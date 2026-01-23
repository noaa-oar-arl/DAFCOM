"""
Aero Protocol Verification Tests for Dataloader.
🍃⚡ Verifies both Eager (NumPy) and Lazy (Dask) backends.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.forecast_mode.dataloader import Load_Observation, Load_Static_Data


def test_load_observation_eager_lazy(tmp_path):
    # Setup dummy AirNow data
    obs_folder = tmp_path / "obs"
    obs_folder.mkdir()
    obs_filename = "test_obs.nc"
    filepath = obs_folder / obs_filename

    # Create synthetic observation data
    # 10 sites, 24 hours
    sites = np.arange(10)
    times = pd.date_range("2023-08-01", periods=24, freq="h")

    rng = np.random.default_rng(42)
    pm25_data = rng.random((10, 24))
    # Add some missing values
    pm25_data[0, 0] = -1

    ds_dummy = xr.Dataset(
        {
            "PM2.5": (["site", "time"], pm25_data),
            "latitude": (["site"], np.linspace(30, 40, 10)),
            "longitude": (["site"], np.linspace(-100, -90, 10)),
        },
        coords={
            "site": sites,
            "time": times,
        },
    )
    ds_dummy.to_netcdf(filepath)

    # Bounds that include everything
    ll_lat, ur_lat = 20, 50
    ll_lon, ur_lon = -120, -80

    # 1. Eager Load (No chunks)
    ds_eager = Load_Observation.extract_airnow_pm25(str(obs_folder), obs_filename, ll_lat, ur_lat, ll_lon, ur_lon)
    assert not hasattr(ds_eager.pm25.data, "dask"), "Eager load should not have dask attribute"

    # 2. Lazy Load (With chunks)
    ds_lazy = Load_Observation.extract_airnow_pm25(
        str(obs_folder), obs_filename, ll_lat, ur_lat, ll_lon, ur_lon, chunks={"site": 5}
    )
    assert hasattr(ds_lazy.pm25.data, "dask"), "Lazy load must have dask attribute"

    # 3. Verification: Results must be identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # 4. Content Verification
    # We should have 10 sites and 24 hours
    assert ds_eager.sizes["site"] == 10
    assert ds_eager.sizes["time"] == 24
    # Valid count should be (10 * 24) - 1 = 239
    assert ds_eager.pm25.count() == 239

    assert "latitude" in ds_eager
    assert "longitude" in ds_eager

    print("Load_Observation Eager vs Lazy Verification: PASSED")


def test_load_static_data_eager_lazy(tmp_path):
    # Setup dummy static data
    static_folder = tmp_path / "static"
    static_folder.mkdir()
    filename = "static.nc"
    filepath = static_folder / filename

    lat = np.linspace(30, 40, 5)
    lon = np.linspace(-100, -90, 5)
    rng = np.random.default_rng(42)
    data = rng.random((5, 5))

    ds_dummy = xr.Dataset({"Elevation": (["lat", "lon"], data)}, coords={"lat": lat, "lon": lon})
    ds_dummy.to_netcdf(filepath)

    time_length = 10

    # 1. Test Elevation
    da_static = Load_Static_Data.extract_elevation(str(filepath), "Elevation", time_length)

    assert da_static.dims == ("time", "lat", "lon")
    assert da_static.shape == (time_length, 5, 5)

    # Check that all time slices are the same
    for t in range(time_length):
        np.testing.assert_array_equal(da_static.isel(time=t).values, data)

    print("Load_Static_Data Verification: PASSED")


if __name__ == "__main__":
    pytest.main([__file__])
