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
        self.downsample()

    def downsample(self):
        """
        Some of the videos are 24 FPS, some are 30 FPS. For the 30 FPS, we drop
        every 5th frame
        """
        # Get the number we need to remove using the original shape
        num_to_remove = self.data.shape[0] - 45

        # Drop last few frames so that video is divisible by 5
        if self.data.shape[0] % 5 != 0:
            num_to_remove -= self.data.shape[0] % 5
            self.data = self.data[: -(self.data.shape[0] % 5)]

        # Split video into chunks of 5 frames each - 5 chosen arbitrarily
        self.data = np.split(self.data, int(self.data.shape[0] / 5), axis=0)

        # Iteratively drop frames from each chunk until we've matched lowest frame count (45)
        removed = 0
        i = 0
        while removed < num_to_remove:
            self.data[i] = self.data[i][:-1]
            i = (i + 1) % len(self.data)
            removed += 1

        # Concatenate all the splits)
        self.data = np.concatenate(self.data)

        # Update shape
        self.shape = self.data.shape

    def to_vec(self):
        """
        (num_frames x rows x columns x color_channels) --> 1D
        """
        assert len(self.data.shape) == 4, "data is already in vector form"
        return self.data.flatten()

    def __getitem__(self, idx):
        return self.data[idx]


#    def play(self):
#        """
#        Plays video
#        """
#        # Create a VideoCapture object and read from input file
#        cap = cv2.VideoCapture(self.path)
#
#        # Check if camera opened successfully
#        if (cap.isOpened()== False):
#          print("Error opening video  file")
#
#        # Read until video is completed
#        while(cap.isOpened()):
#
#          # Capture frame-by-frame
#          ret, frame = cap.read()
#          if ret == True:
#
#            # Display the resulting frame
#            cv2.imshow('Frame', frame)
#
#            # Press Q on keyboard to  exit
#            if cv2.waitKey(25) & 0xFF == ord('q'):
#              break
#
#          # Break the loop
#          else:
#            break
#
#        # When everything done, release
#        # the video capture object
#        cap.release()
#
#        # Closes all the frames
#        cv2.destroyAllWindows()


class AlgonautsImages(Dataset):
    """
    PyTorch Dataset wrapper for Algonauts videos
    """

    def __init__(
            self,
            dir_path,
            transform=transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    # ),
                    ]
                ),
            ):
        """
        INPUTS:
            dir_path (str): defines the directory where the mp4 videos are stored
        """
        self.load_videos(dir_path)
        self.transform = transform

    def load_videos(self, dir_path):
        """
        INPUTS:
            dir_path (str): defines the directory where the mp4 videos are stored
        OUTPUTS:
            self.videos (list of video objects)
        """
        # Grab all the file names
        onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

        # Convert each file to Video object
        self.videos = []
        for i, f in enumerate(onlyfiles):
            self.videos.append(Video(dir_path + f))
            if i == 100:
                # Concatenate all the frames together
                self.images = torch.cat([torch.tensor(self.videos[i].data) for i in range(100)])
                break

    def __getitem__(self, idx):
        # Grab image at index
        data = self.images[idx]

        # Apply transformation if applicable
        if self.transform:
            data = np.asarray(self.transform(Image.fromarray(data.numpy())))
            data = torch.tensor(data).permute(2, 0, 1).float()
            data = data.float()
            data -= data.min()
            data /= (data.max() - data.min())
        return data

    def __len__(self):
        return torch.tensor(len(self.videos))
