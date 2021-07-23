import skvideo.io
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from torchvision import transforms
import torchvision.models
import torch
from torchvision.transforms.transforms import Normalize, Resize, ToTensor
import random, os
from glob import glob
from tqdm import tqdm


class Video:
    """
    Wrapper util for dealing with mp4 files in python
    """

    def __init__(self, path):
        """
        INPUTS:
            f_name (str): path to mp4 file
        """
        self.path = path
        self.data = skvideo.io.vread(path)
        self.shape = self.data.shape

    def get_frames(self):
        return np.split(self.data, self.data.shape[0], axis=0)


class AlgonautsImages(Dataset):
    """
    PyTorch Dataset wrapper for Algonauts videos
    """

    def __init__(
        self,
        dir_path,
        num_videos,
        load_single_frames = False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    torch.tensor([0.6096, 0.5945, 0.5317]),
                    torch.tensor([0.2046, 0.2017, 0.2060]),
                ),
            ]
        ),
    ):
        """
        INPUTS:
            dir_path (str): defines the directory where the mp4 videos are stored
        """
        self.num_videos = num_videos
        self.frames = self.load_frames(dir_path, load_single_frames)
        self.transform = transform

    def load_frames(self, dir_path, load_single_frames):
        """
        INPUTS:
            dir_path (str): defines the directory where the mp4 videos are stored
        OUTPUTS:
            self.videos (list of video objects)
        """
        # Grab all the file names
        files = glob(os.path.join(dir_path, "*"))
        self.selected_vids = random.sample(files, self.num_videos)

        frames = []
        if load_single_frames:
            for f in tqdm(self.selected_vids):
                frames += [Video(f).get_frames()[0]]
        else:
            for f in tqdm(self.selected_vids):
                frames += Video(f).get_frames()
        
        return frames

    def __getitem__(self, idx):
        # Grab image at index
        data = self.frames[idx][0, :, :, :]

        # Apply transformation if applicable
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return torch.tensor(len(self.frames))
