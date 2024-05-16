import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import importlib.util
import os
from datetime import datetime
import signal
import torch
import shutil
from configs.pyroot_config import ConfigPaths as project_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument("--config_py", type=str, default="train/annotated/default_train_3d_conv.py",
                        help="Path to the Python configuration file.")
    parser.add_argument("--config_yaml", type=str, default="train/annotated/trainer_conf.yaml",
                        help="Path to the YAML configuration file.")
    parser.add_argument("--override", nargs='*', action='append',
                        help="Override specific configurations in .yaml file. Use the format: key=value, e.g. trainer.max_epochs=10.")
    return parser.parse_args()

def import_module(path):
    """Import a Python module from the given path."""
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_configurations(config_py_path, config_yaml_path):
    """Load configurations from Python and YAML files."""
    hparams = import_module(config_py_path)
    with open(config_yaml_path, 'r') as f:
        training_conf = yaml.safe_load(f)
    return hparams, training_conf

def override_configurations(training_conf, overrides):
    """Override configurations based on user inputs."""
    if overrides:
        for override_arg in overrides:
            for kv in override_arg:
                key, value = kv.split("=")
                print(f"Overriding {key} with {value}")
                keys = key.split(".")
                conf = training_conf
                for k in keys[:-1]:
                    conf = conf.get(k, {})
                conf[keys[-1]] = value
    return training_conf

def set_dirpath_and_save_configs(args, training_conf):
    if training_conf['model_checkpoint_callback']['dirpath'] == None:
        training_conf['model_checkpoint_callback']['dirpath'] = './saves/' + os.path.basename(args.config_py).split('.')[0] + datetime.now().strftime("%Y-%m-%d_%H-%M")
        training_conf['wandb']['name'] = os.path.basename(args.config_py).split('.')[0] + datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(training_conf['model_checkpoint_callback']['dirpath'], exist_ok=True)
    with open(training_conf['model_checkpoint_callback']['dirpath'] + '/config.yaml', 'w') as f:
        yaml.dump(training_conf, f)
    save_path = os.path.join(training_conf['model_checkpoint_callback']['dirpath'], 'config.py')
    shutil.copyfile(args.config_py, save_path)

def cleaning_procedure(signum, frame):
    if signum == signal.SIGINT:
        print("Initiating cleanup procedure...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")


if __name__ == "__main__":  
    '''
    Loads the config.py file containing initializations of the model, dataset, and other hyperparameters.
    '''
    
    args = parse_args()

    config_py_path = project_paths.configs_root / args.config_py
    config_yaml_path = project_paths.configs_root / args.config_yaml
    hparams, training_conf = load_configurations(config_py_path, 
                                                 config_yaml_path)
    training_conf = override_configurations(training_conf, args.override)
    
    if training_conf['model_checkpoint_callback']['dirpath'] == None: #if dirpath is not set, set it to the relative path of the config.py file
        training_conf['model_checkpoint_callback']['dirpath'] = project_paths.save_root / args.config_py[:-3]

    training_conf['wandb']['name'] = args.config_py[:-3]

    #save config
    os.makedirs(training_conf['model_checkpoint_callback']['dirpath'], exist_ok=True)
    
    save_path = training_conf['model_checkpoint_callback']['dirpath'] / 'config.py'
    shutil.copyfile(hparams.config_paths.configs_root / args.config_py, save_path)
    
    with open(training_conf['model_checkpoint_callback']['dirpath'] / 'config.yaml', 'w') as f:
        training_conf['model_checkpoint_callback']['dirpath'] = str(training_conf['model_checkpoint_callback']['dirpath'])
        yaml.dump(training_conf, f)
        
    #logger = None
    if training_conf['meta']['logging'] == 'wandb':
        logger = WandbLogger(**training_conf['wandb'])
    else:
        logger = None
    
    callbacks = []
    model_checkpoint_callback = ModelCheckpoint(**training_conf['model_checkpoint_callback'],
                                     filename='-{step}-{epoch:02d}-{' +
                                        training_conf['model_checkpoint_callback']['monitor'] + ':.4f}',)
    callbacks.append(model_checkpoint_callback)
    callbacks.append(EarlyStopping(**training_conf['early_stopping_callback']))
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    trainer = pl.Trainer(**training_conf['trainer'], logger=logger, callbacks=callbacks)

    model = hparams.model
    train_loader = hparams.train_loader
    val_loader = hparams.val_loader
    test_loader = hparams.test_loader

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    #save best model as best.ckpt
    best_model_path = model_checkpoint_callback.best_model_path
    shutil.copyfile(best_model_path, os.path.join(training_conf['model_checkpoint_callback']['dirpath'], 'best.ckpt'))  
