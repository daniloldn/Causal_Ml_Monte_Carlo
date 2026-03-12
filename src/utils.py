import yaml
from pathlib import Path

PATH = Path("configs")

def load_config(name:str):

    config_dir = PATH / f"{name}.yaml"

    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)
        
    return config