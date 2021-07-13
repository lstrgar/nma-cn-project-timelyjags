import os
import sys
import random
import time
import torch
import importlib
import argparse
import yaml
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib.models import *
from lib.preprocess import *
from lib.postprocess import *
from datasets.algonauts_video import algonauts_video

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add path to look for dataset object
sys.path.append('./datasets/')

def parse_args():
	"""
	Command line argument parser
	"""
	parser = argparse.ArgumentParser(description = 'Train Trajectory Variational Autoencoder')
	parser.add_argument('--model_config', dest='model_config',
						help = 'path to yaml file containing the configurations for the model',
						type = str, required=True)

	parser.add_argument('--train_config', dest='train_config',
						help = 'path to yaml file containing the configurations for training',
						type = str, required=True)

	args = parser.parse_args()
	return args

def yaml2dict(args):
	"""
	Helper function to convert configuration files passed from the command line 
	to dictionaries
	"""
	args = vars(args)
	dicts = []
	for key in args.keys():
		if args[key] == None:
			dicts.append(None)
		else:
			with open(args[key]) as f:
				dicts.append(yaml.load(f, Loader=yaml.FullLoader))
	return dicts	

def train(model, train_loader, optimizer, device, epoch):
    # Keep track of the average loss over the epoch
    total_loss = 0
    start_time = time.time()
    for i, x in enumerate(train_loader):
        # Reset gradients 
        optimizer.zero_grad()

        # Send current data point to device
        x = x.to(device)

        # Convert seq to a float32 instead of a double - maybe change this?
        x = x.float()
        y = x

        # Gather the desired output states
        out_states = y

        # Calculate the desired output actions
        out_actions = y[:,1:,:] - y[:,:-1,:]

        # Perform a forward pass - seq = [batch_size x seq_len x input_size]
        nll, kld = model.forward(x, out_states, out_actions, i)
        loss = nll + kld

        if i == 1000:
            model.forward(x, out_states, out_actions, i)

        # Backpropagation and calculate the gradient
        loss.backward()

        # Clip gradients
        clip = 5
        torch.nn.utils.clip_grad_norm(model.parameters(),clip)

        # Update weights
        optimizer.step()

        # Add to the total loss
        total_loss += loss.item()

        # Print progress
        if i % 1000 == 0 and i != 0:
            end_time = time.time()
            t = end_time - start_time
            start_time = end_time
            print('Epoch '+str(epoch)+': '+
                str(i)+' of '+
                str(len(train_loader))+
                ' samples processed - loss: '+str(total_loss / i)+
                ' ('+str(t)+' secs)')

	# Compute Average loss
    avg_loss = total_loss / i
    return avg_loss

def main():
	# Parse arguments
    args = parse_args()

	# Convert the yaml files to dictionaries
    model_config, train_config = yaml2dict(args)

    # Instantiate training dataset
    train_data = algonauts_video(train_config['data_path'])
	
	# Preprocess the data
    # train_data = preprocess(train_data)

	# Create corresponding DataLoaders
    train_loader = DataLoader(train_data,
		train_config['batch_size'],
		shuffle = True,
		num_workers=2)

	# Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Create model
    model = GlobalHiddenTrajectoryModel(model_config, train_data)
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    for epoch in range(train_config['num_epochs']):
        # Train for 1 epoch
        epoch_loss = train(model, train_loader, optimizer, device, epoch)	

        # Avg loss over epoch
        epoch_loss /= len(train_loader)

        # Print current loss every 10 epochs
        print('Epoch: {}/{}.............'.format(epoch, train_config['num_epochs']), end=' ')
        print("Loss: {:.4f}".format(epoch_loss))

        # Save the trained model
        torch.save(model.state_dict(), train_config['save_name']+'.pt')

if __name__ == "__main__":
	main()
