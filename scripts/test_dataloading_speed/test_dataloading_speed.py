import configs.pyroot_config as pyroot_config
from torch.utils.data import default_collate
from src.data.mice_dataset_factory import create_dataset, trails_split
from pytorch_lightning.callbacks import *
import time

config_paths = pyroot_config.ConfigPaths()

### DATASET

data_dir = '/scratch/giuliano-nerfcomputing/deepsniff-processed/'

train_mice_names = ['Lausanne']
test_mice_names = ['Edirne']
validation_size = 0.2

train_trails, val_trails = trails_split(root_dir=data_dir,
                                        mice_names=train_mice_names,
                                        validation_size=validation_size)

window_size = 5         # Must be odd number
signal_window_size = 1  # Must be odd number


t_start = time.time()

train_dataset = create_dataset(root_dir=data_dir,
                                mice_names=train_mice_names,
                                trails_to_load=train_trails,
                                window_size=window_size,
                                signal_window_size=signal_window_size,
                                #transforms=transforms,
                                load_in_memory=True)   

t_end = time.time()
#print root dir
print(f"Root_dir: {data_dir}")
print(f"Time to load dataset: {t_end - t_start}")


# Root_dir: /scratch-local/users/markoc-haeslerlab/deepsniff-processed
# Time to load dataset: 172s


# Root_dir: /scratch/markoc-haeslerlab/deepsniff-processed
# Time to load dataset: 390s

# Root_dir: /scratch/giuliano-nerfcomputing/deepsniff-processed
# Time to load dataset: 390s


# Root_dir: /mnt/haeslerlab/haeslerlabwip2024/Users/marko/deepsniff-processed
# Time to load dataset: ~2100s