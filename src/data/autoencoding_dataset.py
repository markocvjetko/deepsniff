import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tt
from pathlib import Path
from tqdm import tqdm
import os
#tt v2
import torchvision.transforms.v2 as tt

class AutoencoderDataset(Dataset):
    """
    The expect the following structure.
    root_dir
    ├── trail_1
    │   ├── frames_dir
    │   │   ├── 0.png
    │   │   ├── 1.png
    │   │   ├── ...
    ├── trail_2
    │   ├── ...
    ├── ...

    """
    def __init__(self, root_dir, frames_dir, transforms=None):  # Add this line
        
        self.root_dir = Path(root_dir)
        self.frames_dir = frames_dir
        self.transforms = transforms
    
        #get all subdirs in root_dir
        self.trails = [d for d in self.root_dir.iterdir() if d.is_dir()]

        #get paths to all frame dirs
        self.frame_paths = [trail / self.frames_dir for trail in self.trails]
        
        self.trail_lengths = [len(list(frame_path.glob('*.png'))) for frame_path in self.frame_paths]
        self.cumulative_lengths = np.cumsum(self.trail_lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        trail_index = np.argmax(self.cumulative_lengths > index)
        if trail_index > 0:
            frame_index = index - self.cumulative_lengths[trail_index - 1]
        else:
            frame_index = index

        frame_path = self.frame_paths[trail_index] / f'{frame_index}.png'
        frame = Image.open(frame_path)

        if self.transforms is not None:
            frame = self.transforms(frame) 
            
        return frame


def main():

    dataset = AutoencoderDataset(root_dir='/scratch-local/users/markoc-haeslerlab/sniff-pretrain',
                                frames_dir='cropped_frames')
    
    print(len(dataset))
    print(dataset[0].size)
    print()
    #save images to /mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain-test2
    #os.mkdir('/mnt/haeslerlab/haeslerlabwip2024/Users/marko/sniff-pretrain-test2', )
    for i in tqdm(range(100000)):
        frame = dataset[i]
        if i % 5000 == 0:
            print(frame.shape)


if __name__ == "__main__":
    main()