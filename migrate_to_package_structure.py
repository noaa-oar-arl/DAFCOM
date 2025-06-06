#!/usr/bin/env python3
"""
DAFCOM Migration Script

This script helps migrate from the old project structure to the new package structure.
It copies necessary files and updates import statements.

Usage:
    python migrate_to_package_structure.py

Author: GitHub Copilot
Date: June 6, 2025
"""

import os
import sys
import shutil
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dafcom.migration")

# Get project root directory
project_root = Path(__file__).parent

# Source and destination directories
src_dir = project_root / "src" / "forecast_mode"
dst_dir = project_root / "src" / "dafcom"

# Ensure destination directories exist
os.makedirs(dst_dir / "forecast" / "models", exist_ok=True)
os.makedirs(dst_dir / "forecast" / "processor", exist_ok=True)
os.makedirs(dst_dir / "forecast" / "examples", exist_ok=True)
os.makedirs(dst_dir / "utils", exist_ok=True)

# Files to copy with their new locations
files_to_copy = [
    # ML models
    (src_dir / "ml_models" / "model_factory.py", dst_dir / "forecast" / "models" / "model_factory.py"),
    (src_dir / "ml_models" / "time_series_data.py", dst_dir / "forecast" / "models" / "time_series_data.py"),
    (src_dir / "ml_models" / "train_forecasting_model.py", dst_dir / "forecast" / "models" / "training.py"),

    # XGBoost files
    (src_dir / "step6_ML_Xgboost" / "step3_xgboost_training.py", dst_dir / "forecast" / "models" / "xgboost" / "xgboost_training.py"),
    (src_dir / "step7_apply_Xgboost" / "step4_xgboost_apply.py", dst_dir / "forecast" / "models" / "xgboost" / "xgboost_apply.py"),

    # Unified processor
    (src_dir / "unified_training_processor.py", dst_dir / "forecast" / "processor" / "unified_processor.py"),

    # Examples
    (src_dir / "examples" / "train_forecasting_models.py", dst_dir / "forecast" / "examples" / "train_forecasting_models.py"),

    # Config files
    (src_dir / "ml_models" / "config" / "lstm_model_config.yaml", dst_dir / "forecast" / "models" / "config" / "lstm_model_config.yaml"),
    (src_dir / "ml_models" / "config" / "transformer_model_config.yaml", dst_dir / "forecast" / "models" / "config" / "transformer_model_config.yaml"),
]

def update_imports(file_path):
    """Update import statements in a file to use the new package structure."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace import statements
    content = re.sub(
        r'from ml_models\.([a-zA-Z0-9_]+) import',
        r'from dafcom.forecast.models.\1 import',
        content
    )
    content = re.sub(
        r'from unified_training_processor import',
        r'from dafcom.forecast.processor.unified_processor import',
        content
    )
    content = re.sub(
        r'import unified_training_processor',
        r'import dafcom.forecast.processor.unified_processor as unified_training_processor',
        content
    )

    with open(file_path, 'w') as f:
        f.write(content)


def main():
    """Main migration function."""
    logger.info("Starting DAFCOM migration to package structure")

    # Copy files
    for src_file, dst_file in files_to_copy:
        if src_file.exists():
            # Create destination directory if it doesn't exist
            os.makedirs(dst_file.parent, exist_ok=True)

            # Copy file
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied {src_file} to {dst_file}")

            # Update imports in the copied file
            if dst_file.suffix == ".py":
                update_imports(dst_file)
                logger.info(f"Updated imports in {dst_file}")
        else:
            logger.warning(f"Source file not found: {src_file}")

    # Install the package in development mode
    logger.info("Installing DAFCOM package in development mode")
    os.system(f"{sys.executable} -m pip install -e {project_root}")

    logger.info("Migration completed successfully!")
    logger.info("\nYou can now use DAFCOM as a proper Python package:")
    logger.info("  from dafcom.forecast import models, processor")
    logger.info("  from dafcom.utils import config\n")


if __name__ == "__main__":
    main()
