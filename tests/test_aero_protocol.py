import numpy as np
import pytest
import xarray as xr

from src.forecast_mode.dataloader import Load_AOD, Load_Meteo


def test_backend_agnostic_aod(tmp_path):
    # Create dummy directory structure
    aod_folder = tmp_path / "aod"
    year, month, day = "2023", "08", "01"
    day_dir = aod_folder / year / f"{year}{month}" / f"{year}{month}{day}"
    day_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)

    # Create a dummy NetCDF file
    file_path = day_dir / "aqm.t12z.phy.f001"
    ds_dummy = xr.Dataset(
        {
            "aod": (["time", "level", "lat", "lon"], rng.random((1, 1, 10, 10))),
        },
        coords={
            "lat": np.linspace(20, 50, 10),
            "lon": np.linspace(-130, -60, 10),
            "time": [np.datetime64("2023-08-01T12:00:00")],
        },
    )
    ds_dummy.to_netcdf(file_path)

    # Test MFDataset (returns Dask by default in recent xarray)
    ds_mf = Load_AOD.extract_data(str(aod_folder), [year], [month], [day], 10, 10)
    assert hasattr(ds_mf.aod.data, "dask"), "open_mfdataset should be lazy by default"

    # Test with explicit chunks
    ds_lazy = Load_AOD.extract_data(str(aod_folder), [year], [month], [day], 10, 10, chunks={"time": 1})
    assert hasattr(ds_lazy.aod.data, "dask")

    # Assert values are identical
    xr.testing.assert_allclose(ds_mf.compute(), ds_lazy.compute())

    print("AOD Verification: PASSED")


def test_meteo_evaporation(tmp_path):
    # Create dummy directory structure
    meteo_folder = tmp_path / "meteo"
    year, month, day = "2023", "08", "01"
    day_dir = meteo_folder / year / f"{year}{month}" / f"{year}{month}{day}"
    day_dir.mkdir(parents=True)

    # Create a dummy NetCDF file with evaporation components
    file_path = day_dir / "aqm.t12z.phy.f001"
    ds_dummy = xr.Dataset(
        {
            "evbs_ave": (["time", "level", "lat", "lon"], np.ones((1, 1, 10, 10))),
            "evcw_ave": (["time", "level", "lat", "lon"], np.ones((1, 1, 10, 10))),
        },
        coords={
            "lat": np.linspace(20, 50, 10),
            "lon": np.linspace(-130, -60, 10),
            "time": [np.datetime64("2023-08-01T12:00:00")],
        },
    )
    ds_dummy.to_netcdf(file_path)

    ds = Load_Meteo.extract_data(str(meteo_folder), "evaporation", [year], [month], [day], 10, 10)

    # Check if evaporation is the sum of components
    # 1 + 1 = 2
    assert np.all(ds.evaporation.compute() == 2)
    assert hasattr(ds.evaporation.data, "dask")

    print("Meteo Evaporation Verification: PASSED")


if __name__ == "__main__":
    pytest.main([__file__])
