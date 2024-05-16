import torch
from dataset.mice_dataset import MouseSniffingVideoDatasetMultipleFramesLabeled
from dataset.dataset_factory import create_dataset, trails_split
from dataset.transforms import init_transforms
import torchvision.transforms.v2 as tt
from models.lightning_module import DeepSniff
from models.models import MobileNetV3
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *



transforms_val = tt.Compose([tt.Normalize(mean=[0.36], std=[0.2])])


window_size = 5         # Must be odd number
signal_window_size = 5  # Must be odd number

root_dir = '/scratch/markoc-haeslerlab/deepsniff-processed/'
train_mice_names = ['Lausanne', 'Neuchatel', 'Montreux']
test_mice_names = ['Edirne']
validation_size = 0.2


val_dataset = create_dataset(root_dir=root_dir,
                                mice_names=train_mice_names,
                                transforms=transforms_val,
                                window_size=window_size,
                                signal_window_size=signal_window_size)

loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

model = DeepSniff.load_from_checkpoint('saves/default_train_3_mice/-step=6290-epoch=16-val_loss=0.1394.ckpt')


for batch in loader:
    x, y = batch
    x = x.cuda()
    pred = model(x)
    print(pred, y)
    break