import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tt
from pathlib import Path
from tqdm import tqdm
class MouseSniffingVideoDatasetMultipleFramesLabeled(Dataset):

    def __init__(self, root_dir, 
                 video_path='frames', 
                 signal_path='signal.txt', 
                 window_size=9, 
                 signal_window_size=5, 
                 loading_transforms=tt.Compose([tt.PILToTensor(), tt.CenterCrop((128, 128)),]), # "tt.Resize((224, 224), antialias=True)" removed
                 transforms=None, 
                 target_transforms=None,
                 load_in_memory=True):  # Add this line
                 
        self.root_dir = Path(root_dir)
        self.video_path = self.root_dir / video_path
        self.signal_path = self.root_dir / signal_path
        self.loading_transforms = loading_transforms
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.window_size = window_size
        self.signal_window_size = signal_window_size
        self.load_in_memory = load_in_memory

        #get frame paths, sort them and load them into a tensor
        self.frame_paths = sorted(self.video_path.glob('*'), key=lambda x: int(x.stem))
        if self.load_in_memory:
            self.frames = torch.stack([self.load_frame_(frame_path) for frame_path in self.frame_paths])
        else:
            self.frames = None
        self.signal = self.load_signal_(self.signal_path)

    def load_frame_(self, frame_path):
        with Image.open(frame_path) as frame:
            if self.loading_transforms:
                frame = self.loading_transforms(frame)
                #print(frame.shape)
            return frame

    def load_signal_(self, signal_path): 
        try:
            signal = np.loadtxt(signal_path) #error if the signal file doesnt exist
        except Exception:
            print(f'Signal file not found: {signal_path}. Using zero signal instead.')
            signal = np.zeros(len(self.frames))     
        points_to_remove = self.window_size - self.signal_window_size
        if points_to_remove > 0:
            signal = signal[points_to_remove // 2:-(points_to_remove // 2)]
        return torch.tensor(signal, dtype=torch.float32)
  
    def __len__(self):
        return len(self.frame_paths) - self.window_size + 1

    def __getitem__(self, index):
        if self.load_in_memory:
            frames = self.frames[index:index+self.window_size]
        else:
            frames = torch.stack([self.load_frame_(frame_path) for frame_path in self.frame_paths[index:index+self.window_size]])

        labels = self.signal[index:index+self.signal_window_size]

        #print(frames.shape)
        if self.transforms: 
            frames = self.transforms(frames)


        if self.target_transforms:
            labels = self.target_transforms(labels)
        
        return frames, labels
    