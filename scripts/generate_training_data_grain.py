
import sys,os
from os import listdir
from os.path import isdir,join
from itertools import groupby as g
import copy
import random

import numpy as np
import cv2
from tqdm import tqdm
from skimage import filters
from scipy import ndimage
#~ from medpy.filter.smoothing import anisotropic_diffusion
#~ from skimage.morphology import skeletonize
from skimage.morphology import thin
from scipy.ndimage.measurements import label
from scipy.spatial import distance
from skimage import morphology
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#~ from mpi4py import MPI
#~ comm = MPI.COMM_WORLD
#~ size = comm.Get_size()
#~ rank = comm.Get_rank()
size=1
rank=0




# make tqdm progress bar compatible with mpi
# by displaying for rank 0 only
def tqdm_list(array, description):
	if (rank == 0 and len(array)>5):
		return tqdm(array, desc=description)
	else:
		return array


#~ create directionary   
def mkdir(dir_name):
    if (os.path.exists(dir_name) == False):
        os.makedirs(dir_name)


#~ GET LIST OF FILES FOR CONVERSION
def getFiles(path,type='png'):
	inputFileList=[]
	for root, dirs, files in os.walk(path):
		for name in files:
			if name.endswith((type)):
				inputFileList.append(join(root,name))
	return inputFileList


def get_frame_name(file_name):
	file_name = str(file_name)
	file_name = os.path.basename(file_name)
	return file_name[:-4]

	
def get_frame_index(frame):
	file_name = get_frame_name(frame)
	tokens = file_name.split('_')
	index = int(tokens[-1])
	return index

def generate_unet_training_data(raw_file_list, binary_file, mask_file,
		training_data_dir, ground_truth_dir, sample_per_img, 
		top_w, bottom_w, left_w, right_w, random_rotate=False,
		random_flip_ud=False, random_flip_lr=False, focus_particle=False, 
		starting_index=0,raw_format='npy', small=False):
	

	if small:
		IMAGE_DIMENSION = 284
		BOARDER = 44
		INNER_DIMENSION = 196
	else:
		IMAGE_DIMENSION = 572
		BOARDER = 92
		INNER_DIMENSION = 388
	num_file = len(raw_file_list)
	binary = cv2.imread(binary_file, 0)
	mask = cv2.imread(mask_file, 0)

			
	for img_idx in tqdm_list(range(num_file),'image'):
		
		if raw_file_list[img_idx][-3:] == 'npy':
			raw = np.load(raw_file_list[img_idx])
		else:
			raw = cv2.imread(raw_file_list[img_idx], 0)
		
	
		for sample_idx in range(sample_per_img):
				raw_s = raw.copy()
				binary_s = binary.copy()		
					
				flag = False
				raw_temp = raw_s
				binary_temp = binary_s
				while not flag:
					top = random.randint(top_w,bottom_w-INNER_DIMENSION-1)-BOARDER
					left = random.randint(left_w,right_w-INNER_DIMENSION-1)-BOARDER
					raw_s = raw_temp[top:top+IMAGE_DIMENSION, left:left+IMAGE_DIMENSION]
					top += BOARDER-top_w
					left += BOARDER-left_w
					binary_s = binary_temp[top:top+INNER_DIMENSION, left:left+INNER_DIMENSION]
					binary_s[binary_s > 125] = 255
					binary_s[binary_s < 255] = 0
					if not focus_particle:
						flag = True
					elif np.amin(mask[top:top+INNER_DIMENSION, left:left+INNER_DIMENSION])>0:
							flag = True
				
				if random_flip_ud and random.choice([True, False]):
					raw_s = np.flipud(raw_s)
					binary_s = np.flipud(binary_s)
				if random_flip_lr and random.choice([True, False]):
					raw_s = np.fliplr(raw_s)
					binary_s = np.fliplr(binary_s)
				if random_rotate and random.choice([True, False]):
					raw_s = np.rot90(raw_s)
					binary_s = np.rot90(binary_s)
				
				file_idx = img_idx*sample_per_img+sample_idx+starting_index
				raw_file = os.path.join(training_data_dir, str(file_idx).zfill(6)+'.'+raw_format)
				binary_file = os.path.join(ground_truth_dir, str(file_idx).zfill(6)+'.png')
				assert raw_s.shape==(IMAGE_DIMENSION,IMAGE_DIMENSION)
				assert binary_s.shape==(INNER_DIMENSION,INNER_DIMENSION)
				if raw_format=='png':
					cv2.imwrite(raw_file, raw_s.astype('uint16'))
				else:
					np.save(raw_file, raw_s)
				cv2.imwrite(binary_file, binary_s)
			

def main():

	sourceDir = '/scratch/utkur/hongwei/image_at_night/grain/2nd_static_movie/'
	binary_file = '/scratch/utkur/hongwei/image_at_night/grain/2nd_static_movie/binary.png'
	mask_file = '/scratch/utkur/hongwei/image_at_night/grain/2nd_static_movie/mask.png'
	saveDir = '/scratch/utkur/hongwei/image_at_night/grain/2nd_static_movie/'
	raw_img_dir = os.path.join(sourceDir, 'drift_corrected_array')
	top = 1008
	bottom = 4008
	left = 416
	right = 4016
	
	sample_per_img = 25

	random_flip_ud = True
	random_flip_lr = True
	random_rotate = True
	

	training_data_dir = os.path.join(saveDir, 'training_images')
	ground_truth_dir = os.path.join(saveDir, 'ground_truth')
	if rank==0:
		mkdir(training_data_dir)
		mkdir(ground_truth_dir)
	#~ comm.Barrier()


	raw_file_list = getFiles(raw_img_dir, 'npy')
	raw_file_list = np.sort(raw_file_list)

	proc_raw_file_list = np.array_split(raw_file_list, size)
	num_frames = raw_file_list.size
	frame_list = range(num_frames)
	proc_frame_list = np.array_split(frame_list, size)
	starting_index = proc_frame_list[rank][0]*sample_per_img
	generate_unet_training_data(proc_raw_file_list[rank], 
		binary_file, mask_file, training_data_dir, ground_truth_dir,
		sample_per_img, top, bottom, left,right, elastic_deform,
		deform_per_img, random_rotate, random_flip_ud,
		random_flip_lr, starting_index)



if __name__ == '__main__':
	main()
	
