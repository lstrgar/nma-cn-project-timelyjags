import torch
import numpy as np
from sklearn.decomposition import PCA
import torchvision.models 
from torchvision import transforms

def resnet_features(data):
	"""
	Extract resnet features from the end of the network, before the classification head
	"""
	feat_extractor = torchvision.models.resnet34(pretrained=True)
	feat_extractor.fc = torch.nn.Identity()
	
	return feat_extractor(data)


def preprocess(data):
	"""
	INPUT:
		data (algonauts_videos): object defined in dataset directory
	output:
		preprocess_data (algonauts_videos): object where the numpy arrays containing the video
			have been preprocessed
	"""
	
	data = resnet_features(data)
	
	# Test fwd pass
	# data = feat_extractor(torch.tensor(data.videos[0].data).permute(0,3,1,2).float())

	return data

if __name__ == "__main__":
	print('something')
