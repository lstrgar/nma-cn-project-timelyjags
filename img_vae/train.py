from models import *
import torch.optim as optim
from algonauts_images import AlgonautsImages
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

### HYPER PARAMETERS ###
latent_dim = 3
epochs = 1 
batch_size = 256
num_videos = 1000
clip = 5
lr = 0.005
M_N = 0.005
########################

model = VanillaVAE(in_channels=3, latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_function = model.loss_function

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)


def train(train_loader):

    for epoch in range(epochs):

        total_loss = 0

        for i, img in enumerate(train_loader):

            if torch.any(torch.isinf(img)) or torch.any(torch.isnan(img)):
                print("INPUTS contain NAN or INF")

            # Reset them gradients
            optimizer.zero_grad()
            img = img.to(device)

            # Forward pass
            ex = model(img)

            if torch.any(torch.isinf(ex[0])) or torch.any(torch.isnan(ex[0])):
                print("OUTPUTS contain NAN or INF")

            loss = loss_function(*ex, M_N=M_N)

            # Backward pass
            loss["loss"].backward()

            # Clip them gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            # Update weights
            optimizer.step()

            # Add to total loss
            total_loss += loss["loss"].item()

        # Compute average loss over the epoch
        print(total_loss / len(imgs))

		# Save current model iteration
        torch.save({
	        "epoch": epoch,
		    "model_state_dict": model.state_dict(),
		    "optimizer_state_dict": optimizer.state_dict(),
		    "loss": loss,
            }, './img_vae.pt')


if __name__ == "__main__":
    imgs = AlgonautsImages(
        dir_path="/home/andrewulmer/neuromatch/Algonauts2021_devkit/participants_data/AlgonautsVideos268_All_30fpsmax/",
        num_videos=num_videos,
    )
    train_loader = DataLoader(imgs, batch_size=batch_size, shuffle=True, num_workers=2)
    train(train_loader)
