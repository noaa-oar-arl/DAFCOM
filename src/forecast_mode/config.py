import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config as a dictionary."""
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
