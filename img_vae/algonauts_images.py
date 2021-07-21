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
from torchvision.transforms.transforms import Resize, ToTensor
import random, os
from glob import glob


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
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        ),
    ):
        """
        INPUTS:
            dir_path (str): defines the directory where the mp4 videos are stored
        """
        self.num_videos = num_videos
        self.frames = self.load_frames(dir_path)
        self.transform = transform

    def load_frames(self, dir_path):
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
        for f in self.selected_vids:
            frames += Video(f).get_frames()
        return frames

    def __getitem__(self, idx):
        # Grab image at index
        data = self.frames[idx][0, :, :, :]

        # Apply transformation if applicable
        if self.transform:
            data = self.transform(data)
            # data -= data.min()
            # data /= data.max() - data.min()
        return data

    def __len__(self):
        return torch.tensor(len(self.frames))
