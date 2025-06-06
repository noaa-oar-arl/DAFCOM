#!/usr/bin/env python3
"""
Unit tests for the unified atmospheric data processing approach.

These tests validate that the unified approach works correctly for
processing different types of atmospheric data.

Author: GitHub Copilot
Date: June 2025
"""

import unittest
import os
import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
import xarray as xr
import yaml

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from modules.parallel_data_loader import ParallelDataLoader


class TestUnifiedProcessing(unittest.TestCase):
    """Test cases for the unified atmospheric data processing approach."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample data and configuration."""
        # Create temporary directories
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = os.path.join(cls.temp_dir, 'data')
        cls.config_dir = os.path.join(cls.temp_dir, 'config')
        cls.output_dir = os.path.join(cls.temp_dir, 'output')

        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.config_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)

        # Create sample data files
        cls._create_sample_data()

        # Create test configuration
        cls.config_file = os.path.join(cls.config_dir, 'test_config.yaml')
        cls._create_test_config()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)

    @classmethod
    def _create_sample_data(cls):
        """Create sample netCDF files with test data."""
        # Create a grid
        lat = np.linspace(25, 49, 10)
        lon = np.linspace(-125, -65, 15)
        time = np.arange(np.datetime64('2025-06-01'), np.datetime64('2025-06-03'), np.timedelta64(6, 'h'))

        # Create directories for sample data
        sample_data_dir = os.path.join(cls.data_dir, '2025', '202506', '20250601')
        os.makedirs(sample_data_dir, exist_ok=True)

        # Create unified file with multiple variable types (chemistry, meteorology, AOD)
        for i, t in enumerate(time):
            # Create dataset with dimensions
            ds = xr.Dataset(
                coords={
                    'time': [t],
                    'lat': lat,
                    'lon': lon
                }
            )

            # Add meteorology variables
            ds['tmp2m'] = xr.DataArray(
                np.random.uniform(260, 310, size=(1, 10, 15)),
                coords={'time': [t], 'lat': lat, 'lon': lon},
                dims=['time', 'lat', 'lon'],
                attrs={'long_name': '2-meter temperature', 'units': 'K'}
            )

            ds['pressfc'] = xr.DataArray(
                np.random.uniform(90000, 105000, size=(1, 10, 15)),
                coords={'time': [t], 'lat': lat, 'lon': lon},
                dims=['time', 'lat', 'lon'],
                attrs={'long_name': 'Surface pressure', 'units': 'Pa'}
            )

            ds['ugrd10m'] = xr.DataArray(
                np.random.uniform(-10, 10, size=(1, 10, 15)),
                coords={'time': [t], 'lat': lat, 'lon': lon},
                dims=['time', 'lat', 'lon'],
                attrs={'long_name': '10-meter u-wind component', 'units': 'm/s'}
            )

            ds['vgrd10m'] = xr.DataArray(
                np.random.uniform(-10, 10, size=(1, 10, 15)),
                coords={'time': [t], 'lat': lat, 'lon': lon},
                dims=['time', 'lat', 'lon'],
                attrs={'long_name': '10-meter v-wind component', 'units': 'm/s'}
            )

            # Add chemistry variables
            ds['PM25_TOT'] = xr.DataArray(
                np.random.lognormal(mean=2, sigma=1, size=(1, 10, 15)),
                coords={'time': [t], 'lat': lat, 'lon': lon},
                dims=['time', 'lat', 'lon'],
                attrs={'long_name': 'PM2.5 total concentration', 'units': 'ug/m^3'}
            )

            ds['o3'] = xr.DataArray(
                np.random.lognormal(mean=3, sigma=0.5, size=(1, 10, 15)),
                coords={'time': [t], 'lat': lat, 'lon': lon},
                dims=['time', 'lat', 'lon'],
                attrs={'long_name': 'Ozone concentration', 'units': 'ppbv'}
            )

            # Add AOD variable
            ds['aod'] = xr.DataArray(
                np.random.lognormal(mean=-1, sigma=0.8, size=(1, 10, 15)),
                coords={'time': [t], 'lat': lat, 'lon': lon},
                dims=['time', 'lat', 'lon'],
                attrs={'long_name': 'Aerosol optical depth at 550nm', 'units': ''}
            )

            # Save the file
            file_path = os.path.join(sample_data_dir, f'aqm.t12z.unified.f{i:02d}')
            ds.to_netcdf(file_path)

    @classmethod
    def _create_test_config(cls):
        """Create a test configuration file."""
        config = {
            'global': {
                'n_workers': 2,
                'chunk_size': {'time': 2, 'lat': 5, 'lon': 5},
                'memory_limit_gb': 2,
                'use_distributed': False
            },
            'input': {
                'data_dir_pattern': os.path.join(cls.data_dir, '{year}', '{year}{month:02d}', '{year}{month:02d}{day:02d}'),
                'time_range': {
                    'years': [2025],
                    'months': [6],
                    'days': [1],
                    'hours_limit': None
                },
                'file_pattern': 'aqm.t12z.*.f*',
                'coordinate_adjustments': {}
            },
            'variables': {
                'atmospheric_data': {
                    # Meteorology variables
                    't2m': {
                        'source_name': 'tmp2m',
                        'units': 'K',
                        'description': '2-meter temperature',
                        'category': 'meteorology',
                        'scaling': {'method': 'standardize', 'params': {}}
                    },
                    'surface_pressure': {
                        'source_name': 'pressfc',
                        'units': 'Pa',
                        'description': 'Surface pressure',
                        'category': 'meteorology',
                        'scaling': {'method': 'standardize', 'params': {}}
                    },
                    'wind_speed': {
                        'source_name': ['ugrd10m', 'vgrd10m'],
                        'operation': 'wind_speed',
                        'units': 'm/s',
                        'description': '10-meter wind speed',
                        'category': 'meteorology',
                        'scaling': {'method': 'standardize', 'params': {}}
                    },
                    # Chemistry variables
                    'pm25': {
                        'source_name': 'PM25_TOT',
                        'units': 'ug/m^3',
                        'description': 'PM2.5 total concentration',
                        'category': 'chemistry',
                        'scaling': {'method': 'log', 'params': {'offset': 0.1}}
                    },
                    'o3': {
                        'source_name': 'o3',
                        'units': 'ppbv',
                        'description': 'Ozone concentration',
                        'category': 'chemistry',
                        'scaling': {'method': 'log', 'params': {'offset': 0.1}}
                    },
                    # AOD variable
                    'aod_550nm': {
                        'source_name': 'aod',
                        'units': '',
                        'description': 'Total AOD @ 550nm',
                        'category': 'aerosol',
                        'scaling': {'method': 'log', 'params': {'offset': 0.001}}
                    }
                }
            },
            'output': {
                'output_dir': cls.output_dir
            },
            'logging': {
                'level': 'WARNING'
            }
        }

        with open(cls.config_file, 'w') as f:
            yaml.dump(config, f)

    def test_process_all_data(self):
        """Test processing all atmospheric data."""
        data_loader = ParallelDataLoader(self.config_file)
        result = data_loader.process_atmospheric_data()

        # Check that we have all expected variables
        self.assertIn('t2m', result)
        self.assertIn('surface_pressure', result)
        self.assertIn('wind_speed', result)
        self.assertIn('pm25', result)
        self.assertIn('o3', result)
        self.assertIn('aod_550nm', result)

        # Check that dimensions are correct
        self.assertEqual(len(result.time), 8)  # 8 time steps (2 days with 6-hour intervals)
        self.assertEqual(len(result.lat), 10)
        self.assertEqual(len(result.lon), 15)

        # Check that variable metadata contains category
        self.assertEqual(result['t2m'].attrs.get('category'), 'meteorology')
        self.assertEqual(result['pm25'].attrs.get('category'), 'chemistry')
        self.assertEqual(result['aod_550nm'].attrs.get('category'), 'aerosol')

    def test_process_by_category(self):
        """Test processing by category."""
        data_loader = ParallelDataLoader(self.config_file)

        # Test meteorology category
        meteo_result = data_loader.process_atmospheric_data(category='meteorology')
        self.assertIn('t2m', meteo_result)
        self.assertIn('surface_pressure', meteo_result)
        self.assertIn('wind_speed', meteo_result)
        self.assertNotIn('pm25', meteo_result)
        self.assertNotIn('o3', meteo_result)
        self.assertNotIn('aod_550nm', meteo_result)

        # Test chemistry category
        chem_result = data_loader.process_atmospheric_data(category='chemistry')
        self.assertNotIn('t2m', chem_result)
        self.assertNotIn('surface_pressure', chem_result)
        self.assertNotIn('wind_speed', chem_result)
        self.assertIn('pm25', chem_result)
        self.assertIn('o3', chem_result)
        self.assertNotIn('aod_550nm', chem_result)

        # Test aerosol category
        aerosol_result = data_loader.process_atmospheric_data(category='aerosol')
        self.assertNotIn('t2m', aerosol_result)
        self.assertNotIn('pm25', aerosol_result)
        self.assertIn('aod_550nm', aerosol_result)

    def test_process_specific_variables(self):
        """Test processing specific variables across categories."""
        data_loader = ParallelDataLoader(self.config_file)

        # Select one variable from each category
        variables = ['t2m', 'pm25', 'aod_550nm']
        result = data_loader.process_atmospheric_data(variables=variables)

        # Check that only selected variables are included
        self.assertIn('t2m', result)
        self.assertIn('pm25', result)
        self.assertIn('aod_550nm', result)
        self.assertNotIn('surface_pressure', result)
        self.assertNotIn('o3', result)

        # Check that they have the correct categories
        self.assertEqual(result['t2m'].attrs.get('category'), 'meteorology')
        self.assertEqual(result['pm25'].attrs.get('category'), 'chemistry')
        self.assertEqual(result['aod_550nm'].attrs.get('category'), 'aerosol')

    def test_backward_compatibility(self):
        """Test backward compatibility with legacy methods."""
        data_loader = ParallelDataLoader(self.config_file)

        # Test the legacy process_data_type method
        meteo_result = data_loader.process_data_type('meteorology')
        self.assertIn('t2m', meteo_result)
        self.assertIn('surface_pressure', meteo_result)
        self.assertIn('wind_speed', meteo_result)
        self.assertNotIn('pm25', meteo_result)

        # Test the legacy process_all_data_types method
        all_result = data_loader.process_all_data_types()
        self.assertIn('t2m', all_result)
        self.assertIn('pm25', all_result)
        self.assertIn('aod_550nm', all_result)

    def test_derived_variable(self):
        """Test processing a derived variable (wind speed)."""
        data_loader = ParallelDataLoader(self.config_file)

        # Process only the wind speed variable which is derived from u and v
        result = data_loader.process_atmospheric_data(variables=['wind_speed'])

        # Check that the derived variable exists
        self.assertIn('wind_speed', result)

        # Check the values (should be positive)
        wind_values = result['wind_speed'].values
        self.assertTrue(np.all(wind_values >= 0), "Wind speed should be non-negative")


if __name__ == '__main__':
    unittest.main()
