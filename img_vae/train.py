from models import *
import torch.optim as optim
from algonauts_images import AlgonautsImages
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

### HYPER PARAMETERS ###
latent_size = 3
epochs = 200
########################

model = VanillaVAE(3, latent_size)
optimizer = optim.Adam(model.parameters(), lr=0.005)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_loader):
    for epoch in range(epochs):
        total_loss = 0
        for i, img in enumerate(train_loader):

            # Reset them gradients
            optimizer.zero_grad()

            # Forward pass
            ex = model(img)
            loss = model.loss_function(*ex, M_N=0)

            # Backward pass
            loss['loss'].backward()

            # Update weights
            optimizer.step()

            # Add to total loss
            total_loss += loss['loss'].item()

            if i == 2:
                break

        # Compute average loss over the epoch
        print(total_loss / len(imgs))


if __name__ == "__main__":
    dir_path = "/home/andrew_work/neuromatch/Algonauts2021_devkit/participants_data/AlgonautsVideos268_All_30fpsmax/"
    imgs = AlgonautsImages(dir_path)
    train_loader = DataLoader(
            imgs, 1, shuffle=False, num_workers=2
    )
    train(train_loader)
