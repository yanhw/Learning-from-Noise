
import os, sys
import logging
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
#~ import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
#~ from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,optimizers
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, BatchNormalization, Cropping2D
from tensorflow.python.keras.utils.data_utils import Sequence as Sequence
from tqdm import tqdm

############################################################
# PLOT THE VALIDATION AND TEST LOSS AND ACCURACY FOR THE FITTED MODELS
############################################################
def plotMetrics(fileName,history):
    
    lossTrain,accuracyTrain = history.history['loss'],history.history['accuracy']
    lossTest,accuracyTest = history.history['val_loss'],history.history['val_accuracy']
    x = range(1,len(lossTrain)+1)
    #~ print('ccccccccc')
    #~ fig = plt.figure(figsize=(8,4))
    #~ ax1 = fig.add_subplot(121)
    #~ ax2 = fig.add_subplot(122)
    #~ ax1.plot(x,lossTrain,label='train')
    #~ ax1.plot(x,lossTest,label='test')
    #~ ax1.set_xlabel('Iterations')
    #~ ax1.set_ylabel('Loss')
    #~ ax1.set_xlim(min(x),max(x))
    #~ ax1.set_ylim(min(min(lossTrain),min(lossTest)),max(max(lossTrain),max(lossTest)))
    #~ ax1.legend()
    #~ ax2.plot(x,accuracyTrain,label='train')
    #~ ax2.plot(x,accuracyTest,label='test')
    #~ ax2.set_xlabel('Iterations')
    #~ ax2.set_ylabel('Accuracy')
    #~ ax2.set_xlim(min(x),max(x))
    #~ ax2.set_ylim(min(min(accuracyTrain),min(accuracyTest)),max(max(accuracyTrain),max(accuracyTest)))
    #~ ax2.legend()
    #~ plt.savefig(fileName,format='png')
    #~ plt.close()
    #~ print('ccccccccc')
    f = open(fileName+'.dat','w')
    f.write(str(history.history))
    f.close()
############################################################



# initial dim: 284-196
def build_model_small(input_layer, start_neurons):
	# 284 -> 140
	conv1 = Conv2D(start_neurons*1, (3,3), activation="relu", padding="same")(input_layer)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(start_neurons*1, (3,3), activation="relu", padding="same")(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D((2,2))(conv1)
	pool1 = Dropout(0.25)(pool1)
	pool1 = Cropping2D(cropping=((1,1), (1,1)))(pool1)
	
	# 140 -> 68
	conv2 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D((2,2))(conv2)
	pool2 = Dropout(0.5)(pool2)
	pool2 = Cropping2D(cropping=((1,1), (1,1)))(pool2)
	
	# 68 -> 32
	conv3 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D((2,2))(conv3)
	pool3 = Dropout(0.5)(pool3)
	pool3 = Cropping2D(cropping=((1,1), (1,1)))(pool3)
	
    # Middle 32 -> 28
	convm = Conv2D(start_neurons*8, (3,3), activation="relu", padding="same")(pool3)
	convm = BatchNormalization()(convm)
	convm = Conv2D(start_neurons*8, (3,3), activation="relu", padding="same")(convm)
	convm = BatchNormalization()(convm)
	convm = Cropping2D(cropping=((2,2), (2,2)))(convm)
	
    # 28 -> 52
	deconv4 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding="same")(convm)
	conv4 = Cropping2D(cropping=((6,6), (6,6)))(conv3)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Dropout(0.5)(uconv4)
	uconv4 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(uconv4)
	uconv4 = BatchNormalization()(uconv4)
	uconv4 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(uconv4)
	uconv4 = BatchNormalization()(uconv4)
	uconv4 = Cropping2D(cropping=((2,2), (2,2)))(uconv4)
	
    # 90 -> 86
	deconv3 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding="same")(uconv4)
	conv3 = Cropping2D(cropping=((18,18), (18,18)))(conv2)
	uconv3 = concatenate([deconv3, conv3])
	uconv3 = Dropout(0.5)(uconv3)
	uconv3 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Cropping2D(cropping=((2,2), (2,2)))(uconv3)
	
    # 86 -> 168
	deconv2 = Conv2DTranspose(start_neurons*1, (3,3), strides=(2,2), padding="same")(uconv3)
	conv2 = Cropping2D(cropping=((42,42), (42,42)))(conv1)
	uconv2 = concatenate([deconv2, conv2])
	uconv1 = Dropout(0.5)(uconv2)
	uconv1 = Conv2D(start_neurons*1, (3,3), activation="relu", padding="same")(uconv1)
	uconv1 = BatchNormalization()(uconv1)
	uconv1 = Cropping2D(cropping=((1,1), (1,1)))(uconv1)
	uconv1 = Conv2D(start_neurons*1, (3, 3), activation="relu", padding="same")(uconv1)
	uconv1 = BatchNormalization()(uconv1)
	uconv1 = Cropping2D(cropping=((1,1), (1,1)))(uconv1)
    #uconv1 = Dropout(0.5)(uconv1)
	output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
	
	return output_layer


# initial dim: 572
def build_model(input_layer, start_neurons):
	# 572 -> 284
	conv1 = Conv2D(start_neurons*1, (3,3), activation="relu", padding="same")(input_layer)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(start_neurons*1, (3,3), activation="relu", padding="same")(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D((2,2))(conv1)
	pool1 = Dropout(0.25)(pool1)
	pool1 = Cropping2D(cropping=((1,1), (1,1)))(pool1)
	
	# 284 -> 140
	conv2 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D((2,2))(conv2)
	pool2 = Dropout(0.5)(pool2)
	pool2 = Cropping2D(cropping=((1,1), (1,1)))(pool2)
	
	# 140 -> 68
	conv3 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D((2,2))(conv3)
	pool3 = Dropout(0.5)(pool3)
	pool3 = Cropping2D(cropping=((1,1), (1,1)))(pool3)
	
    # 68 -> 32
	conv4 = Conv2D(start_neurons*8, (3,3), activation="relu", padding="same")(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(start_neurons*8, (3,3), activation="relu", padding="same")(conv4)
	conv4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D((2,2))(conv4)
	pool4 = Dropout(0.5)(pool4)
	pool4 = Cropping2D(cropping=((1,1), (1,1)))(pool4)
	
    # Middle 32 -> 28
	convm = Conv2D(start_neurons*16, (3,3), activation="relu", padding="same")(pool4)
	convm = BatchNormalization()(convm)
	convm = Conv2D(start_neurons*16, (3,3), activation="relu", padding="same")(convm)
	convm = BatchNormalization()(convm)
	convm = Cropping2D(cropping=((2,2), (2,2)))(convm)
	
    # 28 -> 52
	deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding="same")(convm)
	conv4 = Cropping2D(cropping=((6,6), (6,6)))(conv4)
	uconv4 = concatenate([deconv4, conv4])
	uconv4 = Dropout(0.5)(uconv4)
	uconv4 = Conv2D(start_neurons*8, (3,3), activation="relu", padding="same")(uconv4)
	uconv4 = BatchNormalization()(uconv4)
	uconv4 = Conv2D(start_neurons*8, (3,3), activation="relu", padding="same")(uconv4)
	uconv4 = BatchNormalization()(uconv4)
	uconv4 = Cropping2D(cropping=((2,2), (2,2)))(uconv4)
	
    # 52 -> 100
	deconv3 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding="same")(uconv4)
	conv3 = Cropping2D(cropping=((18,18), (18,18)))(conv3)
	uconv3 = concatenate([deconv3, conv3])
	uconv3 = Dropout(0.5)(uconv3)
	uconv3 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Conv2D(start_neurons*4, (3,3), activation="relu", padding="same")(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Cropping2D(cropping=((2,2), (2,2)))(uconv3)
	
    # 100 -> 196
	deconv2 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding="same")(uconv3)
	conv2 = Cropping2D(cropping=((42,42), (42,42)))(conv2)
	uconv2 = concatenate([deconv2, conv2])
	uconv2 = Dropout(0.5)(uconv2)
	uconv2 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(uconv2)
	uconv2 = BatchNormalization()(uconv2)
	uconv2 = Conv2D(start_neurons*2, (3,3), activation="relu", padding="same")(uconv2)
	uconv2 = BatchNormalization()(uconv2)
	uconv2 = Cropping2D(cropping=((2,2), (2,2)))(uconv2)
	
    # 196 -> 388
	deconv1 = Conv2DTranspose(start_neurons*1, (3,3), strides=(2,2), padding="same")(uconv2)
	conv1 = Cropping2D(cropping=((90,90), (90,90)))(conv1)
	uconv1 = concatenate([deconv1, conv1])
	uconv1 = Dropout(0.5)(uconv1)
	uconv1 = Conv2D(start_neurons*1, (3,3), activation="relu", padding="same")(uconv1)
	uconv1 = BatchNormalization()(uconv1)
	uconv1 = Cropping2D(cropping=((1,1), (1,1)))(uconv1)
	uconv1 = Conv2D(start_neurons*1, (3, 3), activation="relu", padding="same")(uconv1)
	uconv1 = BatchNormalization()(uconv1)
	uconv1 = Cropping2D(cropping=((1,1), (1,1)))(uconv1)
	uconv1 = Dropout(0.5)(uconv1)
	output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
	
	return output_layer



def read_data(x_data_file_list, y_data_file_list, raw_format, img_size_target = 572, norm_data=True):
	assert len(x_data_file_list) == len(y_data_file_list)
	assert img_size_target == 572 or img_size_target == 284
	#~ assert img_size_target == 572 or img_size_target == 312
	if img_size_target == 572:
		y_size = 388
	else:
		y_size = 196
		#~ y_size = 168
	x_data = []
	y_data = []
	for idx in tqdm(range(len(x_data_file_list))):
	#~ for idx in range(len(x_data_file_list)):
		if raw_format != 'npy':
			gImg = cv2.imread(x_data_file_list[idx], -1).astype('float32')
		else:
			gImg = np.load(x_data_file_list[idx]).astype('float32')
		if(gImg.shape != (img_size_target,img_size_target)):
			continue
		if not norm_data:
			norm_image = gImg
		else:
			lowLimit,highLimit = np.percentile(gImg,0.1),np.percentile(gImg,99.9)
			gImg[gImg<=lowLimit] = lowLimit
			gImg[gImg>=highLimit] = highLimit
			norm_image = cv2.normalize(gImg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		norm_image = norm_image.reshape(img_size_target, img_size_target, 1)
		x_data.append(norm_image)
		y_data.append(cv2.imread(y_data_file_list[idx], 0))
		
	x_data = np.array(x_data).reshape(-1, img_size_target, img_size_target, 1)
	y_data = np.array(y_data).reshape(-1, y_size, y_size, 1)
	y_data[y_data>0] = 1
	
	return x_data, y_data

	
def mkdir(dirName):
    if (os.path.exists(dirName) == False):
        os.makedirs(dirName)


#~ GET LIST OF FILES FOR CONVERSION
def getFiles(path,type='png'):
	inputFileList=[]
	for root, dirs, files in os.walk(path):
		for name in files:
			if name.endswith((type)):
				inputFileList.append(os.path.join(root,name))
	return inputFileList


def get_file_lists(x_data_dir, y_data_dir, raw_format):
	x_data_file_list = getFiles(x_data_dir, raw_format)
	y_data_file_list = getFiles(y_data_dir, 'png')
	x_data_file_list = np.sort(x_data_file_list)
	y_data_file_list = np.sort(y_data_file_list)
	indices = np.arange(x_data_file_list.shape[0])
	np.random.shuffle(indices)

	x_data_file_list = x_data_file_list[indices]
	y_data_file_list = y_data_file_list[indices]
	return x_data_file_list, y_data_file_list


class Batch_Data_Generator(Sequence) :
	def __init__(self, image_filenames, binary_filenames, batch_size, raw_format, img_size_target=572) :
		self.image_filenames = image_filenames
		self.binary_filenames = binary_filenames
		self.batch_size = batch_size
		self.raw_format = raw_format
		self.img_size_target = img_size_target
    
	def __len__(self) :
		return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
	def __getitem__(self, idx) :
		batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
		batch_y = self.binary_filenames[idx * self.batch_size : (idx+1) * self.batch_size]   
		return read_data(batch_x, batch_y, self.raw_format, self.img_size_target)
		
		
def array_to_tensor(array):
	return tf.convert_to_tensor(array)


def training_unet(x_data_dir, y_data_dir, saveDir,
		model_name='unet_model.h5', epochs=50, batch_size=64, by_batch=False,
		use_seed_model=False, seed_model=None, lr = 0.001, x_test_dir=None,
		y_test_dir=None, raw_format='png', small=False, patience=5,
		early_stopping_patience=16, norm_data=True, train_test_split_ratio=0.33):
			
	
	assert (seed_model is not None) or (not use_seed_model)
	assert lr < 1 and lr > 0
	assert train_test_split_ratio < 1 and train_test_split_ratio > 0
	x_data_file_list, y_data_file_list = get_file_lists(x_data_dir, y_data_dir, raw_format)
	assert len(x_data_file_list)==len(y_data_file_list)
	
	if x_test_dir is not None and y_test_dir is not None:
		x_test_file_list, y_test_file_list = get_file_lists(x_test_dir, y_test_dir, raw_format)
		
		assert len(x_test_file_list)==len(y_test_file_list)
	
	if not small:
		img_size_target = 572
		initial_layer = 64
	else:
		img_size_target = 284
		#~ img_size_target = 312
		initial_layer = 32
	
	if by_batch:
		if x_test_dir is not None and y_test_dir is not None:
			x_train = x_data_file_list
			y_train = y_data_file_list
			x_test = x_test_file_list
			y_test = y_test_file_list
		else:
			x_train, x_test, y_train, y_test = train_test_split(x_data_file_list, y_data_file_list, test_size=train_test_split_ratio, random_state=42)
		training_batch_generator = Batch_Data_Generator(x_train,y_train, batch_size, raw_format, img_size_target)
		testing_batch_generator = Batch_Data_Generator(x_test,y_test, batch_size, raw_format, img_size_target)
		train_size = x_train.shape[0]
		test_size = x_test.shape[0]
	else:
		x_data, y_data = read_data(x_data_file_list, y_data_file_list, raw_format, img_size_target, norm_data)
		if x_test_dir is not None and y_test_dir is not None:
			x_train = x_data
			y_train = y_data
			x_test, y_test = read_data(x_test_file_list, y_test_file_list, raw_format, img_size_target, norm_data)
		else:
			x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=train_test_split_ratio, random_state=42)
			
	x_train = array_to_tensor(x_train)
	y_train = array_to_tensor(y_train)
	x_test = array_to_tensor(x_test)
	y_test = array_to_tensor(y_test)
	if use_seed_model:
		model = keras.models.load_model(seed_model)	
	else:
		input_layer = Input((img_size_target, img_size_target, 1))
		if small:
			output_layer = build_model_small(input_layer, initial_layer)
		else:
			output_layer = build_model(input_layer, initial_layer)
		model = tf.compat.v1.keras.Model(input_layer, output_layer)
	#~ model.summary()
	#~ return
	adam = Adam(learning_rate=lr)
	model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
	
	early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=1)
	reduce_lr = ReduceLROnPlateau(factor=0.1, patience=patience, min_lr=0.0000001, verbose=1)
	model_dir = os.path.join(saveDir,'models')
	mkdir(model_dir)
	
	if by_batch:
		filepath = os.path.join(model_dir,"saved-unet-model-{epoch:02d}-.hdf5")
		callbacks_list = [keras.callbacks.ModelCheckpoint(filepath,monitor='val_accuracy',verbose=0,save_best_only=False,mode='auto',save_freq='epoch'),early_stopping, reduce_lr]
		history = model.fit_generator(generator = training_batch_generator, epochs=epochs,validation_data=testing_batch_generator,validation_steps=int(test_size/batch_size),callbacks=callbacks_list)
	else:
		filepath = os.path.join(model_dir,"saved-unet-model-{epoch:02d}-{val_accuracy:.2f}.hdf5")
		checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
		history = model.fit(x_train, y_train,
						validation_data=(x_test, y_test), 
						epochs=epochs,
						batch_size=batch_size,
						callbacks=[early_stopping, checkpoint, reduce_lr])
	filename = os.path.join(saveDir, 'model_history')
	plotMetrics(filename,history)
	model_file = os.path.join(saveDir, model_name)
	model.save(model_file)
	
	keras.backend.clear_session()
	

def main():
	#~ saveDir = '/scratch/utkur/hongwei/image_at_night/nanorod/benchmarkingK2/dm4/test_13/'
	saveDir = 'G:/image_at_night/test_13/'
	model_name = 'unet_nanorod_model.h5'
	x_data_dir = os.path.join(saveDir, 'training_images')
	y_data_dir = os.path.join(saveDir, 'ground_truth')
	by_batch = False
	use_seed_model = False
	seed_model = os.path.join(saveDir, 'saved-unet-model-04-0.99.hdf5')
	epochs = 40
	batch_size = 64
	
	training_unet(x_data_dir, y_data_dir, saveDir,
		model_name=model_name, epochs=epochs, batch_size=batch_size)

	
if __name__ == '__main__':
	main()
	
