import torch
from torch.utils.data import Dataset

class Combined(Dataset):
	"""
	Pytorch dataset wrapper for combining two dataset objects into one, so 
	that we can use PyTorch's DataLoader
	"""
	def __init__(self, in_dataset, out_dataset):
		"""
		INPUTS:
			in_dataset (Dataset): PyTorch dataset object that holds the input data to the model 
			out_datset (Dataset): PyTorch dataset object that holds the output data to the model
		OUTPUTS:
			self.data: each index is a corresponding input output pair
		"""
		self.in_dataset = in_dataset
		self.out_dataset = out_dataset

	def __len__(self):
		"""
		Returns the length of the dataset, assuming they are the same length!
		"""
		in_length = len(self.in_dataset)
		out_length = len(self.out_dataset)
		
		assert in_length == out_length, "Input dataset and output dataset must be same length"		
	
		return in_length

	def __getitem__(self, idx):
		"""
		INPUTS:
			idx (int): the integer representing the idx of the pair of data points you want
		OUTPUTS:
			seq (tensor): the indexed input output pair
		"""
		in_pt = self.in_dataset[idx]
		out_pt = self.out_dataset[idx]
		
		return (in_pt, out_pt) 
