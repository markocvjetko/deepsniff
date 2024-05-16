import configs.pyroot_config as pyroot_config
import os
from torch.utils.data import default_collate
import torch
from src.data.mice_dataset import MouseSniffingVideoDatasetMultipleFramesLabeled
from src.data.mice_dataset_factory import create_dataset, trails_split
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniff
from models.models import MobileNetV3
from pytorch_lightning.callbacks import *
import random

config_paths = pyroot_config.ConfigPaths()

### TRANSFORMS

loading_transforms=tt.Compose([tt.PILToTensor(),
                        tt.Grayscale(num_output_channels=1),
                        tt.Resize([112, 112] ,antialias=True)])
                        
transforms = tt.Compose([
                        tt.ConvertImageDtype(torch.float16),
                        tt.Normalize(mean=[0.36], std=[0.2])])


transforms_val = tt.Compose([
                            tt.ConvertImageDtype(torch.float16),
                            tt.Normalize(mean=[0.36], std=[0.2])])

### DATASET

data_dir = config_paths.data_processed / 'sniff-training-dataset'

#list all subdirs
trails = [f.path for f in os.scandir(data_dir) if f.is_dir()]
#print(trails)

#train val test 0.9, 0.1, 0.1
trails = random.sample(trails, len(trails))
train_trails = trails[:int(0.8*len(trails))]
val_trails = trails[int(0.8*len(trails)):int(0.9*len(trails))]
test_trails = trails[int(0.9*len(trails)):]

#print len
print(len(train_trails), len(val_trails), len(test_trails))

#print trails
print('train:', train_trails)
print('val:', val_trails)
print('test:', test_trails)

window_size = 5         # Must be odd number
signal_window_size = 1  # Must be odd number

train_datasets = []
for trail in train_trails:
    train_datasets.append(MouseSniffingVideoDatasetMultipleFramesLabeled(root_dir=trail,
                                                                        video_path='cropped_frames',
                                                                        signal_path='breathing_onsets.txt',
                                                                        window_size=window_size,
                                                                        signal_window_size=signal_window_size,
                                                                        transforms=transforms,
                                                                        loading_transforms=loading_transforms,
                                                                        load_in_memory=True))
train_dataset = torch.utils.data.ConcatDataset(train_datasets)

val_datasets = []
for trail in val_trails:
    val_datasets.append(MouseSniffingVideoDatasetMultipleFramesLabeled(root_dir=trail,
                                                                        video_path='cropped_frames',
                                                                        signal_path='breathing_onsets.txt',
                                                                        window_size=window_size,
                                                                        signal_window_size=signal_window_size,
                                                                        transforms=transforms_val,
                                                                        loading_transforms=loading_transforms,
                                                                        load_in_memory=True))
val_dataset = torch.utils.data.ConcatDataset(val_datasets)

test_datasets = []
for trail in test_trails:
    test_datasets.append(MouseSniffingVideoDatasetMultipleFramesLabeled(root_dir=trail,
                                                                        video_path='cropped_frames',
                                                                        signal_path='breathing_onsets.txt',
                                                                        window_size=window_size,
                                                                        signal_window_size=signal_window_size,
                                                                        transforms=transforms_val,
                                                                        loading_transforms=loading_transforms,
                                                                        load_in_memory=True))
test_dataset = torch.utils.data.ConcatDataset(test_datasets)



def collate_fn(batch):
    batch = default_collate(batch)
    batch[0] = batch[0].squeeze(2)
    return batch



## MODEL

weights = None 
n_input_channels = window_size
output_dim = signal_window_size

network = MobileNetV3(n_input_channels=n_input_channels, output_dim=output_dim)
loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]))
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.000003, last_epoch=-1, verbose=True)
model = DeepSniff(network, loss, optimizer, scheduler)

#subsets of datasetes
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=10, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=collate_fn)


sample = next(iter(train_loader))
print(sample[0].shape, sample[1].shape)

print(sample[0])
print(train_dataset[0]) 