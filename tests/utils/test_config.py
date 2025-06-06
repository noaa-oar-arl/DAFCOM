"""Unit tests for DAFCOM utils modules."""

import unittest
import os
import tempfile
import yaml
from pathlib import Path

# Import the modules to test
from dafcom.utils.config import load_config


class TestConfigModule(unittest.TestCase):
    """Test cases for the config module."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "test_config.yaml"

        # Create a test config file
        self.test_config = {
            "model_type": "xgboost",
            "feature_columns": ["feature1", "feature2"],
            "target_column": "target",
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            }
        }

        # Write the test config to disk
        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_load_config(self):
        """Test loading a configuration file."""
        config = load_config(self.config_path)

        # Verify the loaded config matches the original
        self.assertEqual(config["model_type"], self.test_config["model_type"])
        self.assertEqual(config["feature_columns"], self.test_config["feature_columns"])
        self.assertEqual(config["target_column"], self.test_config["target_column"])
        self.assertEqual(config["params"]["n_estimators"], self.test_config["params"]["n_estimators"])
        self.assertEqual(config["params"]["max_depth"], self.test_config["params"]["max_depth"])
        self.assertEqual(config["params"]["learning_rate"], self.test_config["params"]["learning_rate"])

    def test_load_config_nonexistent_file(self):
        """Test loading a non-existent configuration file."""
        nonexistent_path = Path(self.temp_dir.name) / "nonexistent.yaml"

        # Loading a non-existent file should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_config(nonexistent_path)


if __name__ == "__main__":
    unittest.main()
