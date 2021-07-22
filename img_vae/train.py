from torch.autograd import grad
from torch.nn.modules.module import ModuleAttributeError
from models import VanillaVAE
import torch.optim as optim
from algonauts_images import AlgonautsImages
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import random
import numpy as np
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
batch_size = 256
num_videos = 100
num_workers = 8
update_step = 10
grad_clip = None

deterministic = True
w_parallel = True
seed = 5
video_dir_path = "/home/luke/work/nma-cn-project-timelyjags/img_vae/test_videos/"
#####################################
#####################################


def train(model, trainloader):

    try:
        loss_function = model.loss_function
    except ModuleAttributeError:
        loss_function = model.module.loss_function

    for epoch in range(epochs):

        # monitor all three losses
        kld_running_loss = 0.0
        mse_running_loss = 0.0
        total_running_loss = 0.0

        for i, img in enumerate(trainloader):

            # Input to device
            img = img.to(device)

            # Reset them gradients
            optimizer.zero_grad()

            # Forward pass
            ex = model(img)

            # Compute loss
            loss = model.module.loss_function(*ex)

            # Backward pass
            loss["loss"].backward()

            # If clipping gradients
            if grad_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # Update weights
            optimizer.step()

            # Update running losses
            kld_running_loss += loss["KLD"].item()
            mse_running_loss += loss["MSE"].item()
            total_running_loss += loss["loss"].item()

            # Every update_step batches print and reset running loss
            if i % update_step == update_step - 1:
                print(
                    "[%d, %5d] loss: %.3f, kld: %.3f, mse: %.3f"
                    % (
                        epoch + 1,
                        i + 1,
                        total_running_loss / update_step,
                        kld_running_loss / update_step,
                        mse_running_loss / update_step,
                    )
                )
                kld_running_loss = 0.0
                mse_running_loss = 0.0
                total_running_loss = 0.0

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

    # Manually set random seed
    if deterministic:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Load videos, config dataloader
    print("Loading dataset...")
    trainset = AlgonautsImages(dir_path=video_dir_path, num_videos=num_videos,)
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
    print("\nBeginning training...\n")
    train(model=model, trainloader=trainloader)
