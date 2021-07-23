import skvideo.io, numpy as np, random
import torch
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor
from glob import glob
from tqdm import tqdm


class Video:
    """
    Wrapper util mp4 files
    """

    def __init__(self, path):
        """
        INPUTS:
            path (str): path to mp4 file
        """
        self.path = path
        # (F, H, W, C)
        self.data = skvideo.io.vread(path)
        self.shape = self.data.shape

    def get_frames(self):
        # Split and return all frames as a list of (H, W, C)
        return np.split(self.data, self.shape[0], axis=0)


class AlgonautsImages(Dataset):
    """
    PyTorch Dataset wrapper for Algonauts videos
    """

    def __init__(
        self,
        dir_path,
        num_videos,
        video_ids=None,
        load_single_frames=False,
        transform=Compose(
            [
                ToTensor(),
                Resize((224, 224)),
                Normalize(
                    torch.tensor([0.6096, 0.5945, 0.5317]),
                    torch.tensor([0.2046, 0.2017, 0.2060]),
                ),
            ]
        ),
    ):
        """
        INPUTS:
            dir_path (str): defines the directory where the mp4 videos are stored
            num_videos (int): number of videos to load frames from
            video_ids (list[int]): zero-based indices of desired videos to load from (mutually exclusive with num_videos)
            load_single_frames (boolean): if true only load the first frame from each selected video
        """

        assert (
            num_videos or video_ids
        ), "num_videos and video_ids are mutually exclusive. please provide only one"

        self.num_videos = num_videos
        self.video_ids = video_ids
        self.frames = self.load_frames(dir_path, load_single_frames)
        self.transform = transform

    def load_frames(self, dir_path, load_single_frames):
        """
        INPUTS:
            dir_path (str): defines the directory where the mp4 videos are stored
            load_single_frames (boolean): if true load only the first frame from each selected video
        OUTPUTS:
            frames (list of video frames)
        """
        # Grab all the file names
        video_list = glob(dir_path + "*.mp4")
        video_list.sort()

        # Select random subset of videos or by specified indices
        if self.num_videos:
            self.selected_vids = random.sample(video_list, self.num_videos)
        else:
            self.selected_vids = [
                vid_path for i, vid_path in enumerate(video_list) if i in self.video_ids
            ]

        # Load frames
        frames = []
        # Only first frame for each video
        if load_single_frames:
            for f in tqdm(self.selected_vids):
                frames += [Video(f).get_frames()[0]]
        # All frames
        else:
            for f in tqdm(self.selected_vids):
                frames += Video(f).get_frames()

        return frames

    def __getitem__(self, idx):
        # Grab frame at index
        data = self.frames[idx][0, :, :, :]

        # Apply transformation if available
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        # Count number of frames in dataset
        return torch.tensor(len(self.frames))
