#!/usr/bin/env python3
"""
Simple validation script for the DAFCOM DataLoader system.
Tests core functionality without requiring external data files.
"""

import numpy as np
import xarray as xr
import sys
import traceback
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

def test_basic_imports():
    """Test that we can import our modules."""
    print("Testing imports...")
    try:
        from parallel_data_loader import ScalingManager, DataTransforms
        print("✓ Successfully imported core modules")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_data_transforms():
    """Test DataTransforms functionality."""
    print("Testing DataTransforms...")
    try:
        from parallel_data_loader import DataTransforms

        # Test wind speed calculation
        u = xr.DataArray([3, 4, 0], dims=['x'])
        v = xr.DataArray([4, 3, 5], dims=['x'])

        wind_speed = DataTransforms.wind_speed(u, v)
        expected = np.sqrt([3**2 + 4**2, 4**2 + 3**2, 0**2 + 5**2])

        np.testing.assert_array_almost_equal(wind_speed.values, expected)
        print("✓ Wind speed calculation works correctly")
        return True
    except Exception as e:
        print(f"✗ DataTransforms test failed: {e}")
        traceback.print_exc()
        return False

def test_scaling_manager():
    """Test ScalingManager functionality."""
    print("Testing ScalingManager...")
    try:
        from parallel_data_loader import ScalingManager

        config = {'ml_settings': {'save_scalers': False, 'scaler_dir': './test_scalers/'}}
        scaling_manager = ScalingManager(config)

        # Test standardization
        data = xr.DataArray([1, 2, 3, 4, 5], dims=['x'])
        scaling_config = {'method': 'standardize', 'params': {}}

        scaled = scaling_manager.fit_transform_variable(data, 'test_var', scaling_config)

        # Should have mean ~0 and std ~1
        mean_val = float(scaled.mean().values)
        std_val = float(scaled.std().values)

        assert abs(mean_val) < 1e-10, f"Mean should be ~0, got {mean_val}"
        assert abs(std_val - 1.0) < 0.1, f"Std should be ~1, got {std_val}"

        print("✓ Scaling manager works correctly")
        return True
    except Exception as e:
        print(f"✗ ScalingManager test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_structure():
    """Test configuration file structure."""
    print("Testing configuration structure...")
    try:
        import yaml
        config_path = Path("config/data_loader_config.yaml")

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check that essential sections exist
            required_sections = ['global', 'input', 'variables', 'output_grid', 'output']
            for section in required_sections:
                assert section in config, f"Missing required section: {section}"

            print("✓ Configuration file structure is valid")
            return True
        else:
            print("✓ Configuration file not present (expected for this test)")
            return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_xarray_operations():
    """Test xarray operations that our system uses."""
    print("Testing xarray operations...")
    try:
        # Create test dataset
        time = np.arange(5)
        lat = np.linspace(25, 30, 10)
        lon = np.linspace(-120, -115, 15)

        temp_data = 290 + 10 * np.random.random((5, 10, 15))

        ds = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], temp_data),
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })

        # Test interpolation
        new_lat = np.linspace(26, 29, 5)
        new_lon = np.linspace(-119, -116, 8)
        interpolated = ds.interp(lat=new_lat, lon=new_lon, method='linear')

        assert interpolated.sizes['lat'] == 5
        assert interpolated.sizes['lon'] == 8

        # Test slicing
        subset = ds.isel(time=slice(0, 3), lat=slice(2, 8))
        assert subset.sizes['time'] == 3
        assert subset.sizes['lat'] == 6

        print("✓ Xarray operations work correctly")
        return True
    except Exception as e:
        print(f"✗ Xarray operations test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("DAFCOM DataLoader System Validation")
    print("=" * 60)

    tests = [
        test_basic_imports,
        test_data_transforms,
        test_scaling_manager,
        test_configuration_structure,
        test_xarray_operations
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All validation tests passed!")
        print("\nThe DAFCOM DataLoader system is ready to use.")
        print("\nNext steps:")
        print("1. Install additional dependencies if needed:")
        print("   pip install dask distributed scikit-learn")
        print("2. Update configuration file with your data paths")
        print("3. Run examples: python examples/run_examples.py")
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        print("Please check the error messages above and install missing dependencies.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
