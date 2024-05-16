import pyrootutils
from dataclasses import dataclass
from pathlib import Path

#find root
path_project_root = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

# Set the path to the root directory
pyrootutils.set_root(
    path=path_project_root,
    project_root_env_var=True,
    pythonpath=True,
    cwd=True
)

@dataclass 
class ConfigPaths():
    project_root: Path = path_project_root
    data_raw : Path = Path('data/raw')
    data_processed: Path = Path('data/processed')
    data_processed_pretrain: Path = data_processed / 'sniff-pretrain-scratch-local'
    save_root: Path = Path('models')
    configs_root: Path = Path('src/configs')

