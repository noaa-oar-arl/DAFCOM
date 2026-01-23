import sys
from unittest.mock import MagicMock

# Mock xesmf and esmpy before they are imported by src.forecast_mode.utils
mock_xe = MagicMock()
mock_esmf = MagicMock()
sys.modules["xesmf"] = mock_xe
sys.modules["esmpy"] = mock_esmf
sys.modules["ESMF"] = mock_esmf

from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from src.forecast_mode.utils import Interpolator, Regrid  # noqa: E402


def test_interpolator_eager_lazy():
    """
    🍃⚡ Aero Protocol: Verify Interpolator logic on both NumPy (Eager) and Dask (Lazy) backends.
    """
    # Create sample data
    times = [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 1)]
    lats = [10.0, 20.0]
    lons = [100.0, 110.0]

    rng = np.random.default_rng()
    data = rng.random((2, 2, 2))
    ds = xr.Dataset({"temp": (["time", "lat", "lon"], data)}, coords={"time": times, "lat": lats, "lon": lons})

    # Target points
    t_target = datetime(2023, 1, 1, 0, 30)
    lat_target = 15.0
    lon_target = 105.0

    # 1. Eager (NumPy) Test
    itp_eager = Interpolator(ds)
    res_eager = itp_eager.get_itp(t_target, lat_target, lon_target, "temp")

    assert isinstance(res_eager, xr.DataArray)
    assert not np.isnan(res_eager.values)

    # 2. Lazy (Dask) Test
    ds_lazy = ds.chunk({"time": 1, "lat": 1, "lon": 1})
    itp_lazy = Interpolator(ds_lazy)
    res_lazy = itp_lazy.get_itp(t_target, lat_target, lon_target, "temp")

    # In xarray, interp() on dask-backed data returns a dask-backed array
    assert res_lazy.chunks is not None

    # Verify results are identical
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values)


def test_regrid_mocked():
    """
    Verify Regrid function structure and metadata preservation using a mocked Regridder.
    """
    # Setup
    rng = np.random.default_rng()
    ds = xr.Dataset(
        {"pollutant": (["time", "lat", "lon"], rng.random((1, 4, 4)))},
        coords={"time": [datetime(2023, 1, 1)], "lat": [1, 2, 3, 4], "lon": [1, 2, 3, 4]},
    )
    ds.attrs["history"] = "Original history"

    # Mock Regridder output
    # The Regridder should return an object that, when called, returns a DataArray on the target grid.
    mock_output = xr.DataArray(
        rng.random((1, 2, 2)), dims=["time", "lat", "lon"], coords={"time": ds.time, "lat": [1.5, 2.5], "lon": [1.5, 2.5]}
    )

    mock_regridder_inst = MagicMock()
    mock_regridder_inst.return_value = mock_output
    mock_regridder_inst.out_horiz_dims_coords = {"lat": [1.5, 2.5], "lon": [1.5, 2.5]}

    # Call Regrid with mocked regridder
    res = Regrid(ds, "pollutant", regridder=mock_regridder_inst)

    # Assertions
    assert "pollutant" in res.data_vars
    assert res.pollutant.dims == ("time", "latitude", "longitude")
    assert "Original history" in res.attrs["history"]
    assert "Aero Protocol" in res.attrs["history"]
    mock_regridder_inst.assert_called_once()


def test_regrid_no_regridder_mocked():
    """
    Verify Regrid function when regridder is not provided (it should create one).
    """
    rng = np.random.default_rng()
    ds = xr.Dataset(
        {"pollutant": (["time", "lat", "lon"], rng.random((1, 4, 4)))},
        coords={"time": [datetime(2023, 1, 1)], "lat": [1, 2, 3, 4], "lon": [1, 2, 3, 4]},
    )

    # Mock Regridder constructor and call
    mock_regridder_inst = MagicMock()
    mock_regridder_inst.return_value = xr.DataArray(
        rng.random((1, 2, 2)), dims=["time", "lat", "lon"], coords={"time": ds.time, "lat": [10, 20], "lon": [100, 110]}
    )
    mock_xe.Regridder.return_value = mock_regridder_inst

    # Call Regrid without regridder
    res = Regrid(ds, "pollutant", ur_lat=20, ll_lat=10, ur_lon=110, ll_lon=100, resolution=5)

    # Verify Regridder was created with correct params
    mock_xe.Regridder.assert_called_once()
    args, _ = mock_xe.Regridder.call_args
    assert args[0] == ds
    assert "lat" in args[1].coords
    assert "lon" in args[1].coords

    assert "pollutant" in res.data_vars


if __name__ == "__main__":
    pytest.main([__file__])
