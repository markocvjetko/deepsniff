import torch
from dataset.mice_dataset import MouseSniffingVideoDatasetMultipleFramesLabeled
from dataset.dataset_factory import create_dataset, trails_split
from dataset.transforms import init_transforms
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniff
from models.models import CNNModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *

### TRANSFORMS
transforms = tt.Compose([
                        tt.Resize(224, antialias=True),
                        tt.Normalize(mean=[0.36], std=[0.2])])


transforms_val = tt.Compose([
                            #resize to 224x224
                            tt.Resize(224, antialias=True),
                            tt.Normalize(mean=[0.36], std=[0.2])])

### DATASET

root_dir = '/scratch/markoc-haeslerlab/deepsniff-processed/'
#root_dir = '/mnt/haeslerlab/haeslerlabwip2024/Users/marko/deepsniff-processed/'

train_mice_names = ['Lausanne', 'Neuchatel', 'Montreux']
test_mice_names = ['Edirne']
validation_size = 0.2

train_trails, val_trails = trails_split(root_dir=root_dir,
                                        mice_names=train_mice_names,
                                        validation_size=validation_size)

window_size = 5         # Must be odd number
signal_window_size = 1  # Must be odd number

train_dataset = create_dataset(root_dir=root_dir,
                               mice_names=train_mice_names,
                               trails_to_load=train_trails,
                               window_size=window_size,
                               signal_window_size=signal_window_size,
                                transforms=transforms)

val_dataset = create_dataset(root_dir=root_dir,
                                mice_names=train_mice_names,
                                trails_to_load=val_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val)

test_dataset = create_dataset(root_dir=root_dir,
                                mice_names=test_mice_names,
                                trails_to_load=None,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val)


# #subsets of 1000 samples each
# train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
# val_dataset = torch.utils.data.Subset(val_dataset, range(1000))
# test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

## MODEL

weights = None 
n_input_channels = window_size
output_dim = signal_window_size

network = CNNModel(num_classes=output_dim)
loss = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000003, last_epoch=-1, verbose=True)
model = DeepSniff(network, loss, optimizer, scheduler)


batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=10)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

