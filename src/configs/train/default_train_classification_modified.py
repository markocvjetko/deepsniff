import torch
from dataset.dataset_factory import create_dataset, trails_split
from dataset.transforms import RandomPixelwiseAffine
from dataset.target_transforms import MapLabels
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniffClassification
from models.models import MobileNetV3
from pytorch_lightning.callbacks import *


###### CREATING UNMASKED DATASET

### TRANSFORMS
transforms_unmasked = tt.Compose([
                        tt.Resize(224, antialias=True),
                        tt.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2), antialias=True),
                        RandomPixelwiseAffine(weight=0.1, bias=0.05, p=0.5, clip=True),
                        tt.RandomHorizontalFlip(p=0.5),
                        tt.Normalize(mean=[0.36], std=[0.2]),
                        tt.RandomRotation(degrees=15)])


transforms_val_unmasked = tt.Compose([
                            #resize to 224x224
                            tt.Resize(224, antialias=True),
                            tt.Normalize(mean=[0.36], std=[0.2])])
# cast to int and squeeze
target_transforms_unmasked = tt.Compose([
                            tt.Lambda(lambda x: torch.squeeze(x).to(dtype=torch.int64)),
                            ])

### DATASET

root_dir_unmasked = '/scratch/markoc-haeslerlab/deepsniff-processed/'
train_mice_names = ['Lausanne', 'Neuchatel', 'Montreux']
test_mice_names = ['Edirne']
validation_size = 0.2

train_trails, val_trails = trails_split(root_dir=root_dir_unmasked,
                                        mice_names=train_mice_names,
                                        validation_size=validation_size)


signal_path = 'signal_peaks.txt'
window_size = 5         # Must be odd number
signal_window_size = 1  # Must be odd number (should be 1 for clas sification)

train_dataset_unmasked = create_dataset(root_dir=root_dir_unmasked,
                               mice_names=train_mice_names,
                               signal_path=signal_path,
                               trails_to_load=train_trails,
                               window_size=window_size,
                               signal_window_size=signal_window_size,
                                transforms=transforms_unmasked,
                                target_transforms=target_transforms_unmasked)

val_dataset_unmasked = create_dataset(root_dir=root_dir_unmasked,
                                mice_names=train_mice_names,
                               signal_path=signal_path,
                                trails_to_load=val_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_unmasked,
                                target_transforms=target_transforms_unmasked)

test_dataset_unmasked = create_dataset(root_dir=root_dir_unmasked,
                                mice_names=test_mice_names,
                               signal_path=signal_path,
                                trails_to_load=None,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_unmasked,
                                target_transforms=target_transforms_unmasked)

##### CREATING MASKED DATASET

### TRANSFORMS

transforms_masked = tt.Compose([
                        RandomPixelwiseAffine(weight=0.1, bias=0.05, p=0.5, clip=True),
                        tt.Normalize(mean=[0.36], std=[0.2]),
                        tt.CenterCrop(224),
                        tt.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2), antialias=True),
                        tt.RandomHorizontalFlip(p=0.5),
                        tt.RandomRotation(degrees=15)])

target_transforms_masked = target_transforms_unmasked

root_dir_masked = '/scratch/markoc-haeslerlab/deepsniff-processed-masked/'

train_dataset_masked = create_dataset(root_dir=root_dir_masked,
                                mice_names=train_mice_names,
                                signal_path=signal_path,
                                trails_to_load=train_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_masked,
                                target_transforms=target_transforms_masked)

val_dataset_masked = create_dataset(root_dir=root_dir_masked,
                                mice_names=train_mice_names,
                                signal_path=signal_path,
                                trails_to_load=val_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_masked,
                                target_transforms=target_transforms_masked)

test_dataset_masked = create_dataset(root_dir=root_dir_masked,
                                mice_names=test_mice_names,
                                signal_path=signal_path,
                                trails_to_load=None,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                transforms=transforms_val_masked,
                                target_transforms=target_transforms_masked)

train_dataset = torch.utils.data.ConcatDataset([train_dataset_unmasked, train_dataset_masked])
val_dataset = torch.utils.data.ConcatDataset([val_dataset_unmasked, val_dataset_masked])
test_dataset = torch.utils.data.ConcatDataset([test_dataset_unmasked, test_dataset_masked])

#subsets of 2k samples, for debugging
# train_dataset = torch.utils.data.Subset(train_dataset, range(2000))
# val_dataset = torch.utils.data.Subset(val_dataset, range(2000))

## MODEL

weights = None 
n_input_channels = window_size
output_dim = 3 * signal_window_size 

network = MobileNetV3(n_input_channels=n_input_channels, output_dim=output_dim)
loss = torch.nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.AdamW(network.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1.0e-08, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000003, last_epoch=-1, verbose=True)
model = DeepSniffClassification(network, loss, optimizer, scheduler)

batch_size = 256

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)


