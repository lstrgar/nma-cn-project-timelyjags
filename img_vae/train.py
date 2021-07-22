from torch.autograd import grad
from models import *
import torch.optim as optim
from algonauts_images import AlgonautsImages
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import numpy as np
from matplotlib import pyplot as plt

### HYPER PARAMETERS ###
latent_dim = 512
epochs = 500
num_videos = 1
lr = 0.005
update_step = 10
grad_clip = None
batch_size = 1
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

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

model.to(device)


def train(train_loader, beta):

    for epoch in range(epochs):

        running_loss = 0.0

        for i, img in enumerate(train_loader):

            img = img.to(device)

            # Reset them gradients
            optimizer.zero_grad()

            # Forward pass
            ex = model(img)

            recons = torch.clone(ex[0][0])
            recons -= torch.min(recons)
            recons /= torch.max(recons) - torch.min(recons)
            plt.imsave("recons.jpg", recons.permute(1, 2, 0).cpu().detach().numpy())

            loss = loss_function(*ex, beta=beta)

            # Backward pass
            loss["loss"].backward()

            # print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))

            if grad_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # print(torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters()])))

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
    inp = imgs[0]
    inp -= torch.min(inp)
    inp /= torch.max(inp) - torch.min(inp)
    plt.imsave("./input.jpg", inp.permute(1, 2, 0).cpu().numpy())
    c, h, w = imgs[0].size()
    beta = 1 / (c * h * w * batch_size)
    print(beta)
    train_loader = DataLoader(imgs, batch_size=batch_size, shuffle=False, num_workers=2)
    train(train_loader, beta=beta)
