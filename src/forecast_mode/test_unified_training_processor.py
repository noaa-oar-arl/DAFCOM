#!/usr/bin/env python3
# filepath: /Users/l22-n04127-res/Documents/GitHub/DAFCOM/src/forecast_mode/test_unified_training_processor.py
"""
Unit tests for the UnifiedTrainingDataProcessor class

These tests validate the functionality of the UnifiedTrainingDataProcessor
by testing each component with synthetic data.

Author: GitHub Copilot
Date: June 5, 2025
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import the training processor
try:
    from unified_training_processor import UnifiedTrainingDataProcessor
except ImportError:
    print("Error importing UnifiedTrainingDataProcessor")
    sys.exit(1)


class MockObservationDataLoader:
    """Mock class for ObservationDataLoader to use in testing."""

    def process_observations(self, start_date, end_date, species=None, sources=None, bbox=None, local_files=None):
        """Create synthetic observation data for testing."""
        # Create time range
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        dates = pd.date_range(start=start_date, end=end_date, freq='H')

        # Create latitude and longitude grid
        lats = np.linspace(25, 49, 10)
        lons = np.linspace(-125, -65, 15)

        # Create sample data array with 100 observation sites
        site_lats = np.random.choice(lats, 100)
        site_lons = np.random.choice(lons, 100)
        site_ids = [f"site_{i:03d}" for i in range(100)]

        # Create times, latitudes, longitudes dimensions
        times = dates

        # Create data arrays for each species
        data_vars = {}

        for sp in species or ["PM25"]:
            # Generate random data with some temporal pattern
            values = np.random.gamma(2, 10, size=(len(times), len(site_ids)))
            # Add a diurnal pattern
            hour_effect = np.sin(np.pi * np.arange(len(times)) % 24 / 12) * 5
            values = values + hour_effect[:, np.newaxis]
            values = np.clip(values, 0, 100)  # Constrain to reasonable range

            data_vars[sp] = xr.DataArray(
                values,
                dims=["time", "site"],
                coords={
                    "time": times,
                    "site": site_ids
                }
            )

        # Create site coordinates dataset
        site_coords = xr.Dataset(
            {
                "latitude": xr.DataArray(site_lats, dims=["site"], coords={"site": site_ids}),
                "longitude": xr.DataArray(site_lons, dims=["site"], coords={"site": site_ids})
            }
        )

        # Combine data variables and site coordinates
        obs_data = xr.Dataset(data_vars)
        obs_data = obs_data.assign_coords(
            latitude=site_coords.latitude,
            longitude=site_coords.longitude
        )

        return obs_data


class MockParallelDataLoader:
    """Mock class for ParallelDataLoader to use in testing."""

    def process_atmospheric_data(self, category=None, variables=None):
        """Create synthetic model data for testing."""
        # Create time range - default to a week of data
        dates = pd.date_range(start="2025-04-25", end="2025-05-01", freq='H')

        # Create latitude and longitude grid
        lats = np.linspace(25, 49, 20)
        lons = np.linspace(-125, -65, 30)

        # Create coords
        coords = {
            "time": dates,
            "lat": lats,
            "lon": lons
        }

        # Determine which variables to include
        if category == "meteorology":
            var_list = ["t2m", "rh", "wind_speed", "surface_pressure"]
        elif category == "chemistry":
            var_list = ["PM25", "OZONE", "NO2"]
        else:
            # Default to a combination if no category specified
            var_list = ["PM25", "t2m", "rh", "wind_speed"]

        # Filter by specific variables if provided
        if variables:
            var_list = [v for v in var_list if v in variables]

        # Create data variables
        data_vars = {}

        for var in var_list:
            # Generate random data appropriate for this variable
            if var == "PM25":
                values = np.random.gamma(2, 15, size=(len(dates), len(lats), len(lons)))
                # Add a diurnal pattern
                for i in range(len(dates)):
                    hour = dates[i].hour
                    # Highest during afternoon
                    hour_factor = 1 + 0.5 * np.sin(np.pi * (hour - 6) / 12)
                    values[i] = values[i] * hour_factor

            elif var == "t2m":
                # Temperature in Kelvin with realistic patterns
                base_temp = 273 + np.linspace(5, 25, len(lats))[:, np.newaxis]  # Latitude gradient
                values = np.tile(base_temp, (len(dates), 1, len(lons)))
                # Add diurnal cycle
                for i in range(len(dates)):
                    hour = dates[i].hour
                    # Coolest at night, warmest in afternoon
                    hour_factor = np.sin(np.pi * (hour - 6) / 12)
                    values[i] = values[i] + hour_factor * 5

            elif var == "rh":
                # RH between 0-100%
                values = 70 + np.random.normal(0, 15, size=(len(dates), len(lats), len(lons)))
                values = np.clip(values, 0, 100)

            elif var == "wind_speed":
                # Wind speed typically 0-20 m/s
                values = np.random.weibull(2, size=(len(dates), len(lats), len(lons))) * 5

            else:
                # Generic data for other variables
                values = np.random.random(size=(len(dates), len(lats), len(lons))) * 10

            data_vars[var] = xr.DataArray(values, dims=["time", "lat", "lon"], coords=coords)

        # Create the dataset
        return xr.Dataset(data_vars, coords=coords)


class TestUnifiedTrainingDataProcessor(unittest.TestCase):
    """Unit tests for UnifiedTrainingDataProcessor."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

        # Create a minimal config
        config = {
            "logging": {"level": "INFO"},
            "paths": {
                "obs_config": "dummy_path",  # Won't be used with mocks
                "model_config": "dummy_path"  # Won't be used with mocks
            },
            "output": {"output_dir": self.temp_dir},
            "processing": {"chunk_size": 100},
            "observations": {"species": ["PM25"]},
            "quality_control": {"valid_ranges": {"PM25": [0, 500]}},
            "feature_engineering": {"add_time_features": True},
            "training": {
                "target_variable": "PM25",
                "train_test_split": {
                    "test_size": 0.2,
                    "temporal_split": True
                }
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

        # Create the processor with mocked components
        self.processor = UnifiedTrainingDataProcessor(self.config_path)
        self.processor.obs_loader = MockObservationDataLoader()
        self.processor.model_loader = MockParallelDataLoader()

    def test_get_observation_data(self):
        """Test retrieving observation data."""
        obs_data = self.processor.get_observation_data(
            start_date="2025-04-25",
            end_date="2025-05-01",
            species=["PM25"]
        )

        # Check if we got an xarray Dataset
        self.assertIsInstance(obs_data, xr.Dataset)

        # Check if it has the expected variable
        self.assertIn("PM25", obs_data.data_vars)

        # Check if it has the expected dimensions
        self.assertIn("time", obs_data.dims)
        self.assertIn("site", obs_data.dims)

        # Check if it has lat/lon coords
        self.assertIn("latitude", obs_data.coords)
        self.assertIn("longitude", obs_data.coords)

    def test_get_model_data(self):
        """Test retrieving model data."""
        model_data = self.processor.get_model_data(
            start_date="2025-04-25",
            end_date="2025-05-01",
            categories=["chemistry"]
        )

        # Check if we got an xarray Dataset
        self.assertIsInstance(model_data, xr.Dataset)

        # Check if it has the expected dimensions
        self.assertIn("time", model_data.dims)
        self.assertIn("lat", model_data.dims)
        self.assertIn("lon", model_data.dims)

        # Check if we have PM25 in the chemistry category
        self.assertIn("PM25", model_data.data_vars)

    def test_interpolate_model_to_obs_locations(self):
        """Test interpolating model data to observation locations."""
        obs_data = self.processor.get_observation_data(
            start_date="2025-04-25",
            end_date="2025-05-01"
        )

        model_data = self.processor.get_model_data(
            start_date="2025-04-25",
            end_date="2025-05-01"
        )

        merged_df = self.processor.interpolate_model_to_obs_locations(model_data, obs_data)

        # Check that we got a DataFrame
        self.assertIsInstance(merged_df, pd.DataFrame)

        # Check that we have both observation and model data
        self.assertIn("PM25", merged_df.columns)
        self.assertIn("model_PM25", merged_df.columns)

    def test_clean_merged_data(self):
        """Test data cleaning functionality."""
        # Create test data
        dates = pd.date_range(start="2025-04-25", periods=100, freq='H')
        test_data = {
            "PM25": np.random.gamma(2, 10, 100),
            "model_PM25": np.random.gamma(2, 12, 100),
            "t2m": np.random.normal(293, 5, 100),
            "time": dates,
            "site_id": np.random.choice([f"site_{i}" for i in range(10)], 100)
        }

        # Add some out-of-range values
        test_data["PM25"][0] = 1000  # Should be filtered out
        test_data["PM25"][1] = np.nan  # Should be filtered out

        merged_df = pd.DataFrame(test_data)

        cleaned_df = self.processor.clean_merged_data(merged_df)

        # Check that we removed the bad data points
        self.assertTrue(len(cleaned_df) < len(merged_df))

        # Check if we added derived features
        self.assertIn("hour_utc", cleaned_df.columns)
        self.assertIn("bias", cleaned_df.columns)

    def test_split_train_test(self):
        """Test train/test splitting."""
        # Create test data
        dates = pd.date_range(start="2025-04-25", periods=100, freq='H')
        test_data = {
            "PM25": np.random.gamma(2, 10, 100),
            "model_PM25": np.random.gamma(2, 12, 100),
            "time": dates
        }

        df = pd.DataFrame(test_data)

        # Test temporal splitting
        train_df, test_df = self.processor.split_train_test(df, test_size=0.2, temporal_split=True)

        self.assertEqual(len(train_df) + len(test_df), len(df))
        self.assertTrue(train_df.time.max() < test_df.time.min())  # Train before test

        # Test random splitting
        train_df, test_df = self.processor.split_train_test(df, test_size=0.2, temporal_split=False)

        self.assertEqual(len(train_df) + len(test_df), len(df))

    def test_process_training_data(self):
        """Test the full data processing pipeline."""
        try:
            full_df, train_df, test_df = self.processor.process_training_data(
                start_date="2025-04-25",
                end_date="2025-04-30",
                save_output=False  # Don't save files for this test
            )

            # Check that we got all three DataFrames
            self.assertIsInstance(full_df, pd.DataFrame)
            self.assertIsInstance(train_df, pd.DataFrame)
            self.assertIsInstance(test_df, pd.DataFrame)

            # Check basic properties
            self.assertEqual(len(full_df), len(train_df) + len(test_df))
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(test_df), 0)

        except Exception as e:
            self.fail(f"process_training_data raised exception: {e}")

    def tearDown(self):
        """Clean up after the test."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
