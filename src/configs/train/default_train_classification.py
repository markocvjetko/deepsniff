import torch
from dataset.dataset_factory import create_dataset, trails_split
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniffClassification
from models.models import MobileNetV3
from pytorch_lightning.callbacks import *

### TRANSFORMS
transforms = tt.Compose([
                        tt.Resize(224, antialias=True),
                        tt.Normalize(mean=[0.36], std=[0.2])])


transforms_val = tt.Compose([
                            #resize to 224x224
                            tt.Resize(224, antialias=True),
                            tt.Normalize(mean=[0.36], std=[0.2])])
# cast to int and squeeze
target_transforms = tt.Compose([
                            tt.Lambda(lambda x: torch.squeeze(x).to(dtype=torch.int64))
                            ])

### DATASET

root_dir = '/scratch/markoc-haeslerlab/deepsniff-processed/'
train_mice_names = ['Lausanne', 'Neuchatel', 'Montreux']
test_mice_names = ['Edirne']
validation_size = 0.2

train_trails, val_trails = trails_split(root_dir=root_dir,
                                        mice_names=train_mice_names,
                                        validation_size=validation_size)


signal_path = 'signal_peaks.txt'
window_size = 5         # Must be odd number
signal_window_size = 1  # Must be odd number (should be 1 for clas sification)

train_dataset = create_dataset(root_dir=root_dir,
                               mice_names=train_mice_names,
                               signal_path=signal_path,
                               trails_to_load=train_trails,
                               window_size=window_size,
                               signal_window_size=signal_window_size,
                                transforms=transforms,
                                target_transforms=target_transforms)

val_dataset = create_dataset(root_dir=root_dir,
                                mice_names=train_mice_names,
                               signal_path=signal_path,
                                trails_to_load=val_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val,
                                target_transforms=target_transforms)

test_dataset = create_dataset(root_dir=root_dir,
                                mice_names=test_mice_names,
                               signal_path=signal_path,
                                trails_to_load=None,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val,
                                target_transforms=target_transforms)

## MODEL

weights = None 
n_input_channels = window_size
output_dim = 3 * signal_window_size 
network = MobileNetV3(n_input_channels=n_input_channels, output_dim=output_dim)

class_weights = torch.tensor([1.0, 1.0, 1.0])
loss = torch.nn.CrossEntropyLoss(weight=t)
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000003, last_epoch=-1, verbose=True)
model = DeepSniffClassification(network, loss, optimizer, scheduler)

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=10)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

