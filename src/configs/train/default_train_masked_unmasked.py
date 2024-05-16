import torch
from dataset.mice_dataset import MouseSniffingVideoDatasetMultipleFramesLabeled
from dataset.dataset_factory import create_dataset, trails_split
from dataset.transforms import init_transforms
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniff
from models.models import MobileNetV3
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *



##### CREATING MASKED DATASET

### TRANSFORMS
#normalizing needs to be done after cropping, because padding value is hardcoded to 0
transforms_masked = tt.Compose([tt.Normalize(mean=[0.36], std=[0.2]),
                        tt.CenterCrop(224)])
transforms_val_masked = tt.Compose([tt.Normalize(mean=[0.36], std=[0.2]),
                            tt.CenterCrop(224)])

### DATASET
root_dir_masked = '/scratch/markoc-haeslerlab/deepsniff-processed-masked/'
train_mice_names = ['Lausanne', 'Neuchatel', 'Montreux']
test_mice_names = ['Edirne']
validation_size = 0.2

# Splitting the trails into training and validation, only done once because the split is the same for both masked and unmasked dataset
train_trails, val_trails = trails_split(root_dir=root_dir_masked,
                                        mice_names=train_mice_names,
                                        validation_size=validation_size)

window_size = 5         # Must be odd number
signal_window_size = 1  # Must be odd number

train_dataset_masked = create_dataset(root_dir=root_dir_masked,
                                mice_names=train_mice_names,
                                trails_to_load=train_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_masked)

val_dataset_masked = create_dataset(root_dir=root_dir_masked,
                                mice_names=train_mice_names,
                                trails_to_load=val_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_masked)

test_dataset_masked = create_dataset(root_dir=root_dir_masked,
                                mice_names=test_mice_names,
                                trails_to_load=None,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_masked)

##### CREATING UNMASKED DATASET


### TRANSFORMS
transforms_unmasked = tt.Compose([
                        tt.Normalize(mean=[0.36], std=[0.2]),
                        tt.CenterCrop(224)])

transforms_val_unmasked = tt.Compose([
                            tt.Normalize(mean=[0.36], std=[0.2]),
                            tt.CenterCrop(224)])

root_dir_unmasked = '/scratch/markoc-haeslerlab/deepsniff-processed/'

train_dataset_unmasked = create_dataset(root_dir=root_dir_unmasked,
                               mice_names=train_mice_names,
                               trails_to_load=train_trails,
                               window_size=window_size,
                               signal_window_size=signal_window_size,
                                transforms=transforms_unmasked)

val_dataset_unmasked = create_dataset(root_dir=root_dir_unmasked,
                                mice_names=train_mice_names,
                                trails_to_load=val_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_unmasked)

test_dataset_unmasked = create_dataset(root_dir=root_dir_unmasked,
                                mice_names=test_mice_names,
                                trails_to_load=None,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_unmasked)

### MERGED DATASET
train_dataset = torch.utils.data.ConcatDataset([train_dataset_unmasked, train_dataset_masked])
val_dataset = torch.utils.data.ConcatDataset([val_dataset_unmasked, val_dataset_masked])
test_dataset = torch.utils.data.ConcatDataset([test_dataset_unmasked, test_dataset_masked])


### DATALOADER
batch_size = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=10)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

## MODEL
weights = None 
n_input_channels = window_size
output_dim = signal_window_size

network = MobileNetV3(n_input_channels=n_input_channels, output_dim=output_dim)
loss = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000003, last_epoch=-1, verbose=True)
model = DeepSniff(network, loss, optimizer, scheduler)

