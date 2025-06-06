"""
Configuration utilities for DAFCOM

This module provides utilities for loading and validating configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict containing the configuration parameters

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that a configuration dictionary contains all required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys

    Returns:
        True if valid, False otherwise
    """
    for key in required_keys:
        if key not in config:
            return False
    return True
