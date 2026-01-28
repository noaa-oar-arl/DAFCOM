import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

# Mock xregrid and esmpy before they are imported by src.forecast_mode.utils
# if they are not already available.
try:
    import esmpy  # noqa: F401
    import xregrid  # noqa: F401

    REAL_AVAILABLE = True
except ImportError:
    sys.modules["xregrid"] = MagicMock()
    sys.modules["esmpy"] = MagicMock()
    sys.modules["ESMF"] = MagicMock()
    REAL_AVAILABLE = False

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

    # Call Regrid with mocked regridder
    res = Regrid(ds, "pollutant", regridder=mock_regridder_inst)

    # Assertions
    assert "pollutant" in res.data_vars
    assert res.pollutant.dims == ("time", "lat", "lon")
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

    with patch("src.forecast_mode.utils.Regridder") as MockRegridder:
        MockRegridder.return_value = mock_regridder_inst
        # Call Regrid without regridder
        res = Regrid(ds, "pollutant", ur_lat=20, ll_lat=10, ur_lon=110, ll_lon=100, resolution=5)

        # Verify Regridder was created with correct params
        MockRegridder.assert_called_once()
        args, kwargs = MockRegridder.call_args
        assert args[0] == ds
        assert "lat" in args[1].coords
        assert "lon" in args[1].coords
        assert kwargs["method"] == "bilinear"

    assert "pollutant" in res.data_vars


def test_regrid_interpolator_integration():
    """
    🍃⚡ Aero Protocol Integration Test: Verify Regrid output works seamlessly with Interpolator.
    """
    rng = np.random.default_rng()
    # 1. Source Data (2 time steps to avoid scipy 1D interp issues)
    times = [datetime(2023, 8, 1, 0), datetime(2023, 8, 1, 1)]
    ds_src = xr.Dataset(
        {"aod": (["time", "lat", "lon"], rng.random((2, 4, 4)))},
        coords={"time": times, "lat": [40, 41, 42, 43], "lon": [-100, -99, -98, -97]},
    )

    # 2. Mock Regridder for target grid
    mock_output = xr.DataArray(
        rng.random((2, 2, 2)),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [40.5, 41.5], "lon": [-99.5, -98.5]},
    )
    mock_regridder = MagicMock()
    mock_regridder.return_value = mock_output

    # 3. Regrid
    ds_regridded = Regrid(ds_src, "aod", regridder=mock_regridder)

    # 4. Interpolate from regridded data
    itp = Interpolator(ds_regridded)
    # Use a time between the two steps
    res = itp.get_itp(datetime(2023, 8, 1, 0, 30), 41.0, -99.0, "aod")

    assert not np.isnan(res.values)
    assert res.dims == ()  # Scalar output for scalar inputs


@pytest.mark.skipif(not REAL_AVAILABLE, reason="xregrid real not available")
def test_regrid_xregrid_integration():
    """
    🍃⚡ Aero Protocol: Integration test with real xregrid (if available).
    """
    rng = np.random.default_rng()
    ds = xr.Dataset(
        {"pollutant": (["time", "lat", "lon"], rng.random((1, 4, 4)))},
        coords={"time": [datetime(2023, 1, 1)], "lat": [1, 2, 3, 4], "lon": [1, 2, 3, 4]},
    )

    res = Regrid(ds, "pollutant", ur_lat=3.5, ll_lat=1.5, ur_lon=3.5, ll_lon=1.5, resolution=1.0)

    assert "pollutant" in res.data_vars
    assert res.pollutant.shape == (1, 2, 2)
    assert "Aero Protocol" in res.attrs["history"]


def test_interpolator_time_synthesis():
    """
    Verify Interpolator's ability to synthesize time coordinates from metadata.
    """
    rng = np.random.default_rng()
    # Dataset without time coordinate
    ds = xr.Dataset({"pm25": (["time", "lat", "lon"], rng.random((2, 4, 4)))}, coords={"lat": [1, 2, 3, 4], "lon": [1, 2, 3, 4]})

    # Initialize with metadata
    itp = Interpolator(ds, dir_year="2023", dir_month="2023_08", date=1)

    assert "time" in itp.ds.coords
    assert itp.ds.time.values[0] == np.datetime64("2023-08-01T12:00:00")


if __name__ == "__main__":
    pytest.main([__file__])
