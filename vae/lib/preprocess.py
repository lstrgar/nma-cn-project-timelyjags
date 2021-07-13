import torch
import numpy as np
from sklearn.decomposition import PCA

def normalize(data):
	"""
	Normalize the data so that we don't get an overflow error in the loss
	"""
	for i in range(len(data.videos)):
		data.videos[i].data = data.videos[i].data / 255
		
		# Check for some numerically unstable operation
		if np.sum(np.isnan(data.videos[i].data)) > 0:
			print('NaN found in normalization')
	
	return data
	
def img_pca(data):
	"""
	Fits a PCA model to all the images in the dataset and then transforms them
	"""
	# concatenate all the frames into one big numpy array
	cat_vid = data.videos[0].data
	for vid_idx in range(1, len(data.videos)):
		cat_vid = np.concatenate((cat_vid, data.videos[vid_idx].data))
		print('vid idx: ', vid_idx)
	
	print('stop')

	data = data.videosreshape(-1, 268*268)
	pca = PCA(n_components = 20)
	

def preprocess(data):
	"""
	INPUT:
		data (algonauts_videos): object defined in dataset directory
	output:
		preprocess_data (algonauts_videos): object where the numpy arrays containing the video
			have been preprocessed
	"""
	data = normalize(data)
	# data = img_pca(data)


if __name__ == "__main__":
	print('something')
