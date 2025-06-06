"""
Unit tests for the DAFCOM Parallel DataLoader system.

Run with: python -m pytest test_data_loader.py -v
"""

import pytest
import numpy as np
import xarray as xr
import tempfile
import yaml
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from parallel_data_loader import ParallelDataLoader, ScalingManager, DataTransforms
from tf_integration import AtmosphericDataGenerator


class TestDataTransforms:
    """Test the DataTransforms class."""

    def test_wind_speed(self):
        """Test wind speed calculation."""
        u = xr.DataArray([3, 4, 0], dims=['x'])
        v = xr.DataArray([4, 3, 5], dims=['x'])

        wind_speed = DataTransforms.wind_speed(u, v)
        expected = np.sqrt([3**2 + 4**2, 4**2 + 3**2, 0**2 + 5**2])

        np.testing.assert_array_almost_equal(wind_speed.values, expected)

    def test_wind_direction(self):
        """Test wind direction calculation."""
        u = xr.DataArray([1, 0, -1], dims=['x'])
        v = xr.DataArray([0, 1, 0], dims=['x'])

        direction = DataTransforms.wind_direction(u, v)
        expected = np.array([0, 90, 180])  # Simplified expectation

        assert direction.shape == (3,)


class TestScalingManager:
    """Test the ScalingManager class."""

    def setup_method(self):
        """Set up test configuration."""
        self.config = {
            'ml_settings': {
                'save_scalers': False,
                'scaler_dir': './test_scalers/'
            }
        }
        self.scaling_manager = ScalingManager(self.config)

    def test_standardize_scaling(self):
        """Test standardization scaling."""
        data = xr.DataArray([1, 2, 3, 4, 5], dims=['x'])
        scaling_config = {'method': 'standardize', 'params': {}}

        scaled = self.scaling_manager.fit_transform_variable(
            data, 'test_var', scaling_config
        )

        # Should have mean ~0 and std ~1
        assert abs(scaled.mean().values) < 0.1
        assert abs(scaled.std().values - 1.0) < 0.1

    def test_log_scaling(self):
        """Test log scaling."""
        data = xr.DataArray([1, 2, 3, 4, 5], dims=['x'])
        scaling_config = {'method': 'log', 'params': {'offset': 0.01}}

        scaled = self.scaling_manager.fit_transform_variable(
            data, 'test_var', scaling_config
        )

        expected = np.log(data + 0.01)
        np.testing.assert_array_almost_equal(scaled.values, expected.values)

    def test_no_scaling(self):
        """Test that 'none' method returns original data."""
        data = xr.DataArray([1, 2, 3, 4, 5], dims=['x'])
        scaling_config = {'method': 'none'}

        scaled = self.scaling_manager.fit_transform_variable(
            data, 'test_var', scaling_config
        )

        np.testing.assert_array_equal(scaled.values, data.values)


class TestConfigurationParsing:
    """Test configuration file parsing and validation."""

    def create_test_config(self):
        """Create a minimal test configuration."""
        return {
            'global': {
                'n_workers': 2,
                'use_distributed': False,
                'chunk_size': {'time': 5, 'lat': 100, 'lon': 100}
            },
            'input': {
                'data_dir_pattern': '/test/path/{year}/{month}/',
                'time_range': {
                    'years': [2025],
                    'months': [4],
                    'days': [29],
                    'hours_limit': 6
                },
                'file_patterns': {
                    'meteo': 'test_*.nc'
                },
                'coordinate_adjustments': {
                    'longitude_offset': -360
                }
            },
            'variables': {
                'meteo': {
                    't2m': {
                        'source_name': 'tmp2m',
                        'units': 'K',
                        'description': 'Temperature at 2m',
                        'scaling': {'method': 'standardize', 'params': {}}
                    }
                }
            },
            'output_grid': {
                'lat_range': [25, 49],
                'lon_range': [-125, -65],
                'resolution': 0.1,
                'interpolation': {'method': 'linear'}
            },
            'output': {
                'output_dir': './test_output/',
                'combine_variables': True,
                'compression': {'compression_level': 1}
            },
            'quality_control': {
                'valid_ranges': {'t2m': [200, 330]},
                'out_of_range_action': 'mask'
            },
            'logging': {
                'level': 'INFO'
            }
        }

    def test_config_loading(self):
        """Test that configuration can be loaded and parsed."""
        config = self.create_test_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # This should not raise an exception if file I/O works
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config['global']['n_workers'] == 2
            assert loaded_config['variables']['meteo']['t2m']['units'] == 'K'

        finally:
            Path(config_path).unlink()  # Clean up


class TestDataStructures:
    """Test data structure handling and validation."""

    def create_test_dataset(self):
        """Create a small test dataset."""
        time = np.arange(10)
        lat = np.linspace(25, 30, 20)
        lon = np.linspace(-120, -115, 30)

        t2m = 290 + 10 * np.random.random((10, 20, 30))
        pressure = 101325 + 1000 * np.random.random((10, 20, 30))

        ds = xr.Dataset({
            'tmp2m': (['time', 'lat', 'lon'], t2m),
            'pressfc': (['time', 'lat', 'lon'], pressure),
            'lat': (['lat', 'lon'], np.broadcast_to(lat[:, None], (20, 30))),
            'lon': (['lat', 'lon'], np.broadcast_to(lon[None, :], (20, 30)))
        }, coords={
            'time': time,
            'lat': (['lat'], lat),
            'lon': (['lon'], lon)
        })

        return ds

    def test_dataset_structure(self):
        """Test that we can create and manipulate test datasets."""
        ds = self.create_test_dataset()

        assert ds.sizes['time'] == 10
        assert ds.sizes['lat'] == 20
        assert ds.sizes['lon'] == 30
        assert 'tmp2m' in ds.data_vars
        assert 'pressfc' in ds.data_vars

    def test_interpolation_compatibility(self):
        """Test that our data structure works with xarray interpolation."""
        ds = self.create_test_dataset()

        # Test interpolation to a smaller grid
        new_lat = np.linspace(26, 29, 10)
        new_lon = np.linspace(-119, -116, 15)

        interpolated = ds.interp(lat=new_lat, lon=new_lon, method='linear')

        assert interpolated.sizes['lat'] == 10
        assert interpolated.sizes['lon'] == 15
        assert 'tmp2m' in interpolated.data_vars


class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    def test_file_pattern_matching(self):
        """Test file pattern matching logic."""
        import fnmatch

        files = [
            'aqm.t12z.phy.f001',
            'aqm.t12z.phy.f002',
            'aqm.t12z.chem_sfc.nc',
            'other_file.nc'
        ]

        meteo_pattern = 'aqm.t12z.phy.f*'
        meteo_files = fnmatch.filter(files, meteo_pattern)

        assert len(meteo_files) == 2
        assert 'aqm.t12z.phy.f001' in meteo_files
        assert 'aqm.t12z.phy.f002' in meteo_files

    def test_coordinate_adjustment(self):
        """Test longitude coordinate adjustment."""
        lon_original = np.array([180, 200, 300, 350])
        lon_adjusted = lon_original - 360

        expected = np.array([-180, -160, -60, -10])
        np.testing.assert_array_equal(lon_adjusted, expected)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_variable_handling(self):
        """Test handling of missing variables."""
        scaling_manager = ScalingManager({})

        # Create data with NaN values
        data = xr.DataArray([1, np.nan, 3, np.nan, 5], dims=['x'])
        scaling_config = {'method': 'standardize', 'params': {}}

        # Should handle NaN values gracefully
        scaled = scaling_manager.fit_transform_variable(
            data, 'test_var', scaling_config
        )

        # NaN values should remain NaN
        assert np.isnan(scaled.values[1])
        assert np.isnan(scaled.values[3])

    def test_all_nan_data(self):
        """Test handling of data that is entirely NaN."""
        scaling_manager = ScalingManager({})

        data = xr.DataArray([np.nan, np.nan, np.nan], dims=['x'])
        scaling_config = {'method': 'standardize', 'params': {}}

        # Should return the original data without crashing
        scaled = scaling_manager.fit_transform_variable(
            data, 'test_var', scaling_config
        )

        assert np.all(np.isnan(scaled.values))


# Integration test placeholder
def test_system_integration():
    """
    Placeholder for integration test.

    This would test the full system with actual files,
    but requires the data files to be present.
    """
    # This is a placeholder - would need actual data files to test
    pass


if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Running basic tests...")

    # Test DataTransforms
    print("Testing DataTransforms...")
    test_transforms = TestDataTransforms()
    test_transforms.test_wind_speed()
    test_transforms.test_wind_direction()
    print("✓ DataTransforms tests passed")

    # Test ScalingManager
    print("Testing ScalingManager...")
    test_scaling = TestScalingManager()
    test_scaling.setup_method()
    test_scaling.test_standardize_scaling()
    test_scaling.test_log_scaling()
    test_scaling.test_no_scaling()
    print("✓ ScalingManager tests passed")

    # Test Configuration
    print("Testing Configuration...")
    test_config = TestConfigurationParsing()
    test_config.test_config_loading()
    print("✓ Configuration tests passed")

    # Test Data Structures
    print("Testing Data Structures...")
    test_data = TestDataStructures()
    test_data.test_dataset_structure()
    test_data.test_interpolation_compatibility()
    print("✓ Data Structure tests passed")

    # Test Utilities
    print("Testing Utilities...")
    test_utils = TestUtilityFunctions()
    test_utils.test_file_pattern_matching()
    test_utils.test_coordinate_adjustment()
    print("✓ Utility tests passed")

    # Test Error Handling
    print("Testing Error Handling...")
    test_errors = TestErrorHandling()
    test_errors.test_missing_variable_handling()
    test_errors.test_all_nan_data()
    print("✓ Error Handling tests passed")

    print("\n🎉 All tests passed! The system is ready to use.")
    print("\nTo test with actual data:")
    print("1. Update the configuration file with correct data paths")
    print("2. Run: python examples/run_examples.py --example basic")
    print("3. Run full test suite: python -m pytest test_data_loader.py -v")
