import os, sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


def mkdir(dir_name):
    if (os.path.exists(dir_name) == False):
        os.makedirs(dir_name)


def get_files(path,type='png'):
	inputFileList=[]
	for root, dirs, files in os.walk(path):
		for name in files:
			if name.endswith((type)):
				inputFileList.append(os.path.join(root,name))
	return inputFileList


def classify_image(input_image, model, input_size=572, output_size=388):
	
	rows, cols = input_image.shape
	blank = input_image.copy()
	blank[:,:] = 0
	padding = int((input_size-output_size)/2)
	padded =  np.pad(input_image,((padding,padding),(padding,padding)), 'reflect')
	x_list = np.arange(0,cols,output_size)
	y_list = np.arange(0,rows,output_size)
	x_list[-1] = min(x_list[-1],cols-output_size)
	y_list[-1] = min(y_list[-1],rows-output_size)
	data_list = []
	top_list = []
	left_list = []
	result_list = []
	for top in y_list:
		for left in x_list:
			data = padded[top:top+input_size,left:left+input_size].copy()
			data = np.array(data).reshape(input_size, input_size, 1)
			lowLimit,highLimit = np.percentile(data,0.1),np.percentile(data,99.9)
			data[data<=lowLimit] = lowLimit
			data[data>=highLimit] = highLimit
			data = cv2.normalize(data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

			top_list.append(top)
			left_list.append(left)
			input_tensor = tf.convert_to_tensor(np.array(data))
			data_list.append(input_tensor)
	
	data_list = np.array(data_list)
	output_tensor = model.predict(data_list)
	result = output_tensor
	result = 255*result
	result = result.astype('uint8')
	result_list = list(result)
	for (result,top,left) in zip(result_list,top_list,left_list):
		result = result.reshape((output_size, output_size))
		blank[top:top+output_size,left:left+output_size] = result
	return blank
	
	
#
def classify_images(model, img_dir, classified_dir, input_size=572, output_size=388):
	num_file = len(img_dir)
	for idx in tqdm(range(num_file)):
		outfile = os.path.join(classified_dir, str(idx).zfill(6)+'.png')
		#skip if this image already exists
		if os.path.isfile(outfile):
			continue
			
		raw_img = np.load(img_dir[idx])
		rows, cols = raw_img.shape
		#pad the image to minimum output size of model if it is smaller than that
		pad_flag = False
		if rows<output_size or cols<output_size:
			row_pad = max(0,output_size-rows)
			col_pad = max(0,output_size-cols)
			pad_flag = True
			raw_img = np.pad(raw_img,((0,row_pad),(0,col_pad)), 'reflect')
		result = classify_image(raw_img, model)
		if pad_flag:
			result = result[:rows,:cols]
		cv2.imwrite(outfile, result)
		

def main():
	model_file_list = ['Z:/hongwei/image_at_night/nanorod/20211119/test_100/saved-unet-model-70-1.00.hdf5',
						'Z:/hongwei/image_at_night/nanorod/20211119/test_101/unet_nanorod_model.h5',
						'Z:/hongwei/image_at_night/nanorod/20211119/test_102/unet_nanorod_model.h5',
						'Z:/hongwei/image_at_night/nanorod/20211119/test_103/unet_nanorod_model.h5',
						'Z:/hongwei/image_at_night/nanorod/20211119/test_104/unet_nanorod_model.h5']
	model_idx_list = ['100','101','102','103','104']
	movie_dir_list = ['Z:/hongwei/image_at_night/nanorod/20210614/J_1000/37/raw_array',
					'Z:/hongwei/image_at_night/nanorod/20210614/J_0500/38/raw_array',
					'Z:/hongwei/image_at_night/nanorod/20210614/J_0100/39/raw_array',
					'Z:/hongwei/image_at_night/nanorod/20210614/J_0050/40/raw_array',
					'Z:/hongwei/image_at_night/nanorod/20210614/J_0010/41/raw_array',
					'Z:/hongwei/image_at_night/nanorod/20210614/J_0005/42/raw_array']
	base_save_dir_list = ['Z:/hongwei/image_at_night/nanorod/20210614/J_1000/37/classified_test_',
						'Z:/hongwei/image_at_night/nanorod/20210614/J_0500/38/classified_test_',
						'Z:/hongwei/image_at_night/nanorod/20210614/J_0100/39/classified_test_',					
						'Z:/hongwei/image_at_night/nanorod/20210614/J_0050/40/classified_test_',
						'Z:/hongwei/image_at_night/nanorod/20210614/J_0010/41/classified_test_',
						'Z:/hongwei/image_at_night/nanorod/20210614/J_0005/42/classified_test_']	
	for model_idx in tqdm(range(len(model_file_list))):
		
		model_file = model_file_list[model_idx]
		model = keras.models.load_model(model_file)
		print(model_file)
		for movie_idx in range(len(movie_dir_list)):	
			img_dir = movie_dir_list[movie_idx]
			img_dir = get_files(img_dir, 'npy')
			img_dir = np.sort(img_dir)
			classified_dir = base_save_dir_list[movie_idx]+model_idx_list[model_idx]
			mkdir(classified_dir)
			classify_images(model, img_dir, classified_dir)
			
		keras.backend.clear_session()

if __name__ == '__main__':
	main()

	
