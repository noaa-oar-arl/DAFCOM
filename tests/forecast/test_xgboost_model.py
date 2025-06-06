"""Unit tests for the XGBoost model in DAFCOM."""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Import the XGBoost model class
from dafcom.forecast.models.xgboost_model import XGBoostModel


class TestXGBoostModel(unittest.TestCase):
    """Test cases for the XGBoostModel class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a temporary directory for model outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        model_path = Path(self.temp_dir.name)

        # Basic configuration for testing
        self.config = {
            'feature_columns': ['feature1', 'feature2', 'feature3'],
            'target_column': 'target',
            'model_path': str(model_path),
            'validation_split': 0.2,
            'random_state': 42,
            'n_estimators': 10,  # Small number for faster tests
            'early_stopping_rounds': 2,
            'verbose': False
        }

        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 100
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })

        # Initialize the model
        self.model = XGBoostModel(self.config)

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test model initialization."""
        self.assertEqual(self.model.feature_columns, self.config['feature_columns'])
        self.assertEqual(self.model.target_column, self.config['target_column'])
        self.assertEqual(self.model.model_path, self.config['model_path'])
        self.assertIsNone(self.model.model)

    def test_train(self):
        """Test model training with valid data."""
        metrics = self.model.train(self.test_data)

        # Verify training produces expected metrics
        self.assertIn('train_rmse', metrics)
        self.assertIn('val_rmse', metrics)
        self.assertIn('train_mae', metrics)
        self.assertIn('val_mae', metrics)
        self.assertIn('train_r2', metrics)
        self.assertIn('val_r2', metrics)
        self.assertIn('training_time', metrics)
        self.assertIn('feature_importance', metrics)
        self.assertIn('model_path', metrics)

        # Check that model was actually created
        self.assertIsNotNone(self.model.model)

        # Check that model file was created
        model_path = Path(metrics['model_path'])
        self.assertTrue(model_path.exists())

    def test_train_missing_columns(self):
        """Test model training with missing columns."""
        # Create data with missing feature
        bad_data = self.test_data.drop(columns=['feature1'])

        # Training should raise ValueError
        with self.assertRaises(ValueError):
            self.model.train(bad_data)

    def test_train_missing_target(self):
        """Test model training with missing target column."""
        # Create data with missing target
        bad_data = self.test_data.drop(columns=['target'])

        # Training should raise ValueError
        with self.assertRaises(ValueError):
            self.model.train(bad_data)

    def test_predict(self):
        """Test model prediction."""
        # Train the model first
        self.model.train(self.test_data)

        # Test prediction
        X_test = self.test_data[self.config['feature_columns']]
        predictions = self.model.predict(X_test)

        # Verify predictions shape
        self.assertEqual(len(predictions), len(X_test))

    def test_predict_without_training(self):
        """Test prediction without training."""
        # Prediction without training should raise RuntimeError
        X_test = self.test_data[self.config['feature_columns']]
        with self.assertRaises(RuntimeError):
            self.model.predict(X_test)

    def test_save_and_load(self):
        """Test saving and loading model."""
        # Train and save model
        metrics = self.model.train(self.test_data)
        model_path = metrics['model_path']

        # Create a new model instance
        new_model = XGBoostModel(self.config)

        # Load the saved model
        new_model.load(model_path)

        # Test that loaded model can predict
        X_test = self.test_data[self.config['feature_columns']]
        predictions = new_model.predict(X_test)

        # Verify predictions shape
        self.assertEqual(len(predictions), len(X_test))


if __name__ == '__main__':
    unittest.main()
