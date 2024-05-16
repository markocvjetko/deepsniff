import configs.pyroot_config as pyroot_config
from torch.utils.data import default_collate
import torch
from src.data.mice_dataset import MouseSniffingVideoDatasetMultipleFramesLabeled
from src.data.mice_dataset_factory import create_dataset, trails_split
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniff
from models.models import MobileNetV3
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *

config_paths = pyroot_config.ConfigPaths()

### TRANSFORMS
transforms = tt.Compose([
                        tt.Resize(224, antialias=True),
                        tt.ConvertImageDtype(torch.float16),
                        tt.Normalize(mean=[0.36], std=[0.2])])


transforms_val = tt.Compose([
                            #resize to 224x224
                            tt.Resize(224, antialias=True),
                            tt.ConvertImageDtype(torch.float16),
                            tt.Normalize(mean=[0.36], std=[0.2])])

### DATASET

data_dir = config_paths.data_processed / 'deepsniff-processed'

train_mice_names = ['Lausanne', 'Neuchatel', 'Montreux']
test_mice_names = ['Edirne']
validation_size = 0.2

train_trails, val_trails = trails_split(root_dir=data_dir,
                                        mice_names=train_mice_names,
                                        validation_size=validation_size)

window_size = 5         # Must be odd number
signal_window_size = 1  # Must be odd number

train_dataset = create_dataset(root_dir=data_dir,
                                mice_names=train_mice_names,
                                trails_to_load=train_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms,
                                load_in_memory=False)

val_dataset = create_dataset(root_dir=data_dir,
                                mice_names=train_mice_names,
                                trails_to_load=val_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val,
                                load_in_memory=False)

test_dataset = create_dataset(root_dir=data_dir,
                                mice_names=test_mice_names,
                                trails_to_load=None,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val,
                                load_in_memory=False)

def collate_fn(batch):
    batch = default_collate(batch)
    batch[0] = batch[0].squeeze()
    return batch



## MODEL

weights = None 
n_input_channels = window_size
output_dim = signal_window_size

network = MobileNetV3(n_input_channels=n_input_channels, output_dim=output_dim)
loss = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.000003, last_epoch=-1, verbose=True)
model = DeepSniff(network, loss, optimizer, scheduler)

#subsets of datasetes

train_dataset = torch.utils.data.Subset(train_dataset, range(0, 1000))
val_dataset = torch.utils.data.Subset(val_dataset, range(0, 1000))
test_dataset = torch.utils.data.Subset(test_dataset, range(0, 100))

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=10, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=collate_fn)


sample = next(iter(train_loader))
print(sample[0].shape, sample[1].shape)

print(sample[0])
print(train_dataset[0]) 