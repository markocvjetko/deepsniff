from src.data.mice_dataset import MouseSniffingVideoDatasetMultipleFramesLabeled
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

def trails_split(root_dir, mice_names, validation_size=0.2, seed=42):
    print(os.getcwd())
    train_trails = {}
    val_trails = {}
    for mouse in mice_names:
        trails = os.listdir(os.path.join(root_dir, mouse))
        train, val = train_test_split(trails, test_size=validation_size, random_state=seed)
        train_trails[mouse] = train
        val_trails[mouse] = val

    return train_trails, val_trails


def create_dataset(root_dir, mice_names, video_path='frames', signal_path='signal_lowpass.txt', window_size=5, signal_window_size=5, 
                    trails_to_load=None, transforms=None, target_transforms=None, load_in_memory=False, **kwargs):

    datasets = []
    for mouse_name in mice_names:
        mouse_dir = os.path.join(root_dir, mouse_name)

        if trails_to_load:
            trails = trails_to_load[mouse_name]
        else:
            trails = os.listdir(os.path.join(mouse_dir))

        for trail in tqdm(trails, desc=f'Loading {mouse_name} trails'):

            dataset = create_single_trail_dataset(
                                    root_dir=os.path.join(mouse_dir, trail),
                                    video_path=video_path,
                                    signal_path=signal_path,
                                    window_size=window_size,
                                    signal_window_size=signal_window_size,
                                    transforms=transforms,
                                    target_transforms=target_transforms,
                                    load_in_memory=load_in_memory
                                    )

            datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)

def create_single_trail_dataset(root_dir, video_path='frames', signal_path='signal.txt', window_size=5, signal_window_size=5, transforms=None, target_transforms=None, load_in_memory=False, **kwargs):
    #print(config):
    return MouseSniffingVideoDatasetMultipleFramesLabeled(
                                    root_dir=root_dir,
                                    video_path=video_path,
                                    signal_path=signal_path,
                                    window_size=window_size,
                                    signal_window_size=signal_window_size,
                                    transforms=transforms, 
                                    target_transforms=target_transforms,
                                    load_in_memory=load_in_memory
                                    )
