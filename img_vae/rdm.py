from torch.autograd import grad
from models import VanillaVAE
import torch.optim as optim
from algonauts_images import AlgonautsImages
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import random
import numpy as np
from scipy.stats import zscore, pearsonr
from matplotlib import pyplot as plt
from tqdm import tqdm

#####################################
#####################################
latent_dim = 1024
epochs = 1000
log_var_min = -10
log_var_max = 10
temperature = 0.01

lr = 0.005
batch_size = 1 
num_videos = 100
num_workers = 8
update_step = 10
grad_clip = None

deterministic = False
w_parallel = False
seed = 5
plot = True
separate_rdms = True # this will create separate RDMs for each label 
video_dir_path = "/home/andrewulmer/nma-cn-project-timelyjags/img_vae/test_algonauts_images/"
rdm_dir = './rdms/'
labels = torch.load('./labels_dict.pt')
#####################################
#####################################

def get_embeddings(model, trainloader):
	
	embeddings = []

	for i, img in enumerate(tqdm(trainloader)):

		# Input to device
		img = img.to(device)

		# Reset them gradients
		optimizer.zero_grad()

		# Forward pass
		embeddings.append(model(img)[2].detach())

	return torch.cat(embeddings)

def compute_rdm(embeddings):
	# Add key for full RDM
	rdms = {key: None for key in labels.keys()} 
	rdms['full_rdm'] = None
	
	for label in rdms.keys():
		
		if label is 'full_rdm':
			sub_embeddings = embeddings
		else:
			sub_embeddings = embeddings[torch.where(labels[label])] 

		# z-score responses to each stimulus
		zresp = zscore(sub_embeddings, axis=1)

		# Compute RDM
		RDM = 1 - (1/zresp.shape[1]*zresp@zresp.T)
	
		rdms[label] = RDM

		# Save in case we need em later
		torch.save(RDM, '{rdm_dir}{l}.pt'.format(rdm_dir=rdm_dir, l=label))
	
	return rdms 

def plot_rdm(rdms):
	
	for label in rdms.keys():	
		
		# Following plot config from neuro-group 
		plt.xlabel("Video ID"); plt.ylabel("Video ID")
		plt.imshow(rdms[label], cmap="seismic", vmax=2)
		plt.colorbar()
		plt.title('Variational Autoencoder {l} Representational Dissimilarity'.format(l=label))
		plt.savefig('{plt_dir}{l}.png'.format(plt_dir=rdm_dir, l=label))
		plt.clf()
	
	print('\nRDM plots saved to ./rdms/\n')


if __name__ == "__main__":

    # Manually set random seed
    if deterministic:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Load videos, config dataloader
    print("Loading dataset...")
    trainset = AlgonautsImages(
        dir_path=video_dir_path, num_videos=num_videos, load_single_frames=True)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Compute KLD loss scaler from input dims + batch size
    c, h, w = trainset[0].size()
    beta = 1 / (c * h * w * batch_size)

    print("\nBuilding model...")
    # Instantiate network
    model = VanillaVAE(
        in_channels=c,
        latent_dim=latent_dim,
        beta=beta,
        log_var_min=log_var_min,
        log_var_max=log_var_max,
        temperature=temperature,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Parallelize if applicable
    if torch.cuda.device_count() > 1 and w_parallel:
        model = nn.DataParallel(model)

    # Move network to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # leggo
    print("\nComputing embeddings...\n")
    embeddings = get_embeddings(model=model, trainloader=trainloader)
    
	# Get that RDM
    rdms = compute_rdm(embeddings)

	# Plot that RDM
    if plot:
        plot_rdm(rdms)
