import torch
from dataset.dataset_factory import create_dataset, trails_split
from dataset.transforms import init_transforms
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniff
from models.models import resnext3d101
from pytorch_lightning.callbacks import *
import tqdm

### TRANSFORMS
transforms = tt.Compose([
                        tt.RandomHorizontalFlip(p=0.5),
                        tt.RandomRotation(degrees=15),
                        tt.Resize(224, antialias=True),
                        tt.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2), antialias=True),
                        tt.Normalize(mean=[0.36], std=[0.2])])


transforms_val = tt.Compose([
                            tt.Resize(224, antialias=True),
                            tt.Normalize(mean=[0.36], std=[0.2])])

### DATASET

root_dir = '/scratch-local/users/markoc-haeslerlab/deepsniff-processed/'
train_mice_names = ['Lausanne', 'Neuchatel', 'Montreux']
test_mice_names = ['Edirne']
validation_size = 0.2

train_trails, val_trails = trails_split(root_dir=root_dir,
                                        mice_names=train_mice_names,
                                        validation_size=validation_size)

window_size = 16        # Must be odd number
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

## MODEL

weights = '/workspaces/markoc-haeslerlab/kinetics_resnext_101_RGB_16_best.pth'
n_input_channels = window_size
output_dim = signal_window_size

network = resnext3d101(weights=weights)

loss = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00003, last_epoch=-1, verbose=True)
model = DeepSniff(network, loss, optimizer, scheduler)


batch_size = 32

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

val_loader = test_loader


#iterate loader
for i,batch in tqdm.tqdm(train_loader):
    pass