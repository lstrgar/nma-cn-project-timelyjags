from torch.autograd import grad
from models import *
import torch.optim as optim
from algonauts_images import AlgonautsImages
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import numpy as np

### HYPER PARAMETERS ###
latent_dim = 512
epochs = 100
batch_size = 256
num_videos = 100
lr = 0.005
M_N = 0.005
update_step = 10
log_nans = True
grad_clip = 5
deterministic = True
seed = 5
########################

torch.autograd.set_detect_anomaly(True)

if deterministic:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

model = VanillaVAE(in_channels=3, latent_dim=latent_dim)

optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_function = model.loss_function

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)


def train(train_loader):

    for epoch in range(epochs):

        running_loss = 0.0

        for i, img in enumerate(train_loader):

            if log_nans and (
                torch.any(torch.isinf(img)) or torch.any(torch.isnan(img))
            ):
                print("INPUTS are NAN or INF")

            img = img.to(device)

            # Reset them gradients
            optimizer.zero_grad()

            # Forward pass
            ex = model(img)

            if log_nans and (
                torch.any(torch.isinf(ex[0])) or torch.any(torch.isnan(ex[0]))
            ):
                print("OUTPUTS are NAN or INF")

            loss = loss_function(*ex, M_N=M_N)

            # Backward pass
            loss["loss"].backward()

            print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))

            if grad_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))

            # if log_nans:
            #     for name, param in model.named_parameters():
            #         print(name, torch.isfinite(param.grad).all())

            # Update weights
            optimizer.step()

            # Add to total loss
            running_loss += loss["loss"].item()

            if i % update_step == update_step - 1:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / update_step)
                )
                running_loss = 0.0

        # Save current model iteration
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            "./img_vae.pt",
        )


if __name__ == "__main__":
    imgs = AlgonautsImages(
        dir_path="/home/luke/work/nma-cn-project-timelyjags/img_vae/test_videos/",
        num_videos=num_videos,
    )
    train_loader = DataLoader(imgs, batch_size=batch_size, shuffle=False, num_workers=2)
    train(train_loader)
