import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from algonauts_images import AlgonautsImages
from models import VanillaVAE
import os

ckpt_path = "/home/luke/work/nma-cn-project-timelyjags/img_vae/img_vae.pt"
video_dir_path = "/home/luke/work/nma-cn-project-timelyjags/img_vae/test_videos/"
save_dir = "/home/luke/work/nma-cn-project-timelyjags/img_vae/test_outs"
log_var_min = -10
log_var_max = 10
temperature = 0.01
latent_dim = 1024
c = 3
beta = 0.0
w_parallel = False
num_videos = 5

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

model = VanillaVAE(
    in_channels=c,
    latent_dim=latent_dim,
    beta=beta,
    log_var_min=log_var_min,
    log_var_max=log_var_max,
    temperature=temperature,
)

checkpoint = torch.load(ckpt_path)

try:
    model.load_state_dict(checkpoint["model_state_dict"])
except:
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.module

# Move network to available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nModel is on device {device}")

# Parallelize if applicable
if torch.cuda.device_count() > 1 and w_parallel:
    model = nn.DataParallel(model)
    print(f"Model parallelized on {torch.cuda.device_count()} GPUs")

trainset = AlgonautsImages(
    dir_path=video_dir_path, num_videos=num_videos, load_single_frames=True
)

for i, sample in enumerate(trainset):
    sample = torch.unsqueeze(sample, 0)
    sample = sample.to(device)
    recons, inp, mu, logvar = model(sample)
    recons = recons[0].cpu().detach()
    inp = inp[0].cpu().detach()
    inp -= torch.min(inp)
    inp /= torch.max(inp) - torch.min(inp)
    recons -= torch.min(recons)
    recons /= torch.max(recons) - torch.min(recons)
    inp = inp.permute(1, 2, 0).numpy()
    recons = recons.permute(1, 2, 0).numpy()
    plt.imsave(os.path.join(save_dir, "input_" + str(i) + ".jpg"), inp)
    plt.imsave(os.path.join(save_dir, "recons_" + str(i) + ".jpg"), recons)

