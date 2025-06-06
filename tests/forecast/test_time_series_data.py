"""Unit tests for the TimeSeriesData module in DAFCOM."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the module to test
from dafcom.forecast.models.time_series_data import TimeSeriesDataPreparation


class TestTimeSeriesDataPreparation(unittest.TestCase):
    """Test cases for the TimeSeriesDataPreparation class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create synthetic time series data
        np.random.seed(42)
        n_samples = 100
        self.dates = [
            datetime(2025, 1, 1) + timedelta(hours=i)
            for i in range(n_samples)
        ]

        self.data = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "target": np.random.randn(n_samples)
        }, index=pd.DatetimeIndex(self.dates))

        # Parameters for the data preparation
        self.input_features = ["feature1", "feature2"]
        self.target_feature = "target"
        self.sequence_length = 5
        self.forecast_horizon = 1

    def test_init(self):
        """Test initialization of TimeSeriesDataPreparation."""
        data_prep = TimeSeriesDataPreparation(
            self.data,
            self.input_features,
            self.target_feature,
            self.sequence_length,
            self.forecast_horizon
        )

        # Check attributes
        self.assertEqual(data_prep.input_features, self.input_features)
        self.assertEqual(data_prep.target_feature, self.target_feature)
        self.assertEqual(data_prep.sequence_length, self.sequence_length)
        self.assertEqual(data_prep.forecast_horizon, self.forecast_horizon)
        self.assertTrue(data_prep.data.equals(self.data))

    def test_create_sequences(self):
        """Test creating sequences from time series data."""
        data_prep = TimeSeriesDataPreparation(
            self.data,
            self.input_features,
            self.target_feature,
            self.sequence_length,
            self.forecast_horizon
        )

        X, y = data_prep.create_sequences()

        # Check shapes
        expected_samples = len(self.data) - self.sequence_length - self.forecast_horizon + 1
        expected_features = len(self.input_features)

        self.assertEqual(X.shape, (expected_samples, self.sequence_length, expected_features))
        self.assertEqual(y.shape, (expected_samples,))

    def test_split_data(self):
        """Test splitting data into training and testing sets."""
        data_prep = TimeSeriesDataPreparation(
            self.data,
            self.input_features,
            self.target_feature,
            self.sequence_length,
            self.forecast_horizon
        )

        train_ratio = 0.8
        X_train, X_test, y_train, y_test = data_prep.split_data(train_ratio)

        # Calculate expected sizes
        X, y = data_prep.create_sequences()
        expected_train_size = int(len(X) * train_ratio)
        expected_test_size = len(X) - expected_train_size

        # Check shapes
        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)


if __name__ == "__main__":
    unittest.main()
