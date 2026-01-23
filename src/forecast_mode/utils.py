"""
Utility functions for DAFCOM.
🍃⚡ Aero-compliant utilities for regridding and interpolation.
"""

from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np
import xarray as xr
import xesmf as xe


def Regrid(
    ds: xr.Dataset,
    variable_name: str,
    ur_lat: Optional[float] = None,
    ll_lat: Optional[float] = None,
    ur_lon: Optional[float] = None,
    ll_lon: Optional[float] = None,
    resolution: Optional[float] = None,
    interpolation_method: str = "bilinear",
    regridder: Optional[xe.Regridder] = None,
) -> xr.Dataset:
    """
    Regrid a variable from a dataset to a regular lat/lon grid.

    🍃⚡ Aero Protocol: This function is backend-agnostic and preserves Dask laziness.
    It supports reusing an existing xESMF Regridder for performance optimization.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the variable to regrid.
    variable_name : str
        The name of the variable to regrid.
    ur_lat : float, optional
        Upper right latitude. Required if regridder is not provided.
    ll_lat : float, optional
        Lower left latitude. Required if regridder is not provided.
    ur_lon : float, optional
        Upper right longitude. Required if regridder is not provided.
    ll_lon : float, optional
        Lower left longitude. Required if regridder is not provided.
    resolution : float, optional
        The desired resolution in degrees. Required if regridder is not provided.
    interpolation_method : str, default 'bilinear'
        The interpolation method to use (passed to xESMF).
    regridder : xesmf.Regridder, optional
        An existing xESMF Regridder instance to reuse.

    Returns
    -------
    xr.Dataset
        A new dataset with the regridded variable, preserving time coordinates and history.

    Raises
    ------
    ValueError
        If neither coordinates/resolution nor a regridder is provided.
    """
    if regridder is None:
        if any(v is None for v in [ur_lat, ll_lat, ur_lon, ll_lon, resolution]):
            raise ValueError("Coordinates and resolution must be provided if regridder is None.")

        # Define output grid
        lat_coords = np.arange(ur_lat, ll_lat, -resolution)  # type: ignore
        lon_coords = np.arange(ll_lon, ur_lon, resolution)  # type: ignore

        ds_out = xr.Dataset(
            {
                "lat": (["lat"], lat_coords),
                "lon": (["lon"], lon_coords),
            }
        )
        regridder = xe.Regridder(ds, ds_out, interpolation_method)
    else:
        # Extract target coordinates from regridder
        lat_coords = regridder.out_horiz_dims_coords["lat"]
        lon_coords = regridder.out_horiz_dims_coords["lon"]

    # Perform regridding
    # xe.Regridder works with Dask-backed xarray objects.
    dr_out = regridder(ds[variable_name])

    # Create final dataset with regridded data
    # 🍃⚡ Using xarray methods to preserve metadata and avoid .values calls.
    # We use 'lat' and 'lon' as the standard coordinate names.
    final_ds = dr_out.to_dataset(name=variable_name)

    # Update history and preserve original dataset attributes if any
    history = ds.attrs.get("history", "")
    new_entry = f"{datetime.now()}: Regridded {variable_name} using {interpolation_method} via Aero Protocol."
    final_ds.attrs["history"] = f"{history}\n{new_entry}" if history else new_entry

    return final_ds


class Interpolator:
    """
    Class for temporal and spatial interpolation.

    🍃⚡ Aero Protocol: Optimized to use xarray's native interpolation,
    supporting both NumPy and Dask backends efficiently.
    """

    def __init__(
        self,
        ds_model: xr.Dataset,
        dir_year: Optional[str] = None,
        dir_month: Optional[str] = None,
        date: Optional[int] = None,
    ):
        """
        Initialize with a model dataset and optional metadata for time synthesis.

        Parameters
        ----------
        ds_model : xr.Dataset
            The model dataset.
        dir_year : str, optional
            Year string for time synthesis if 'time' coordinate is missing.
        dir_month : str, optional
            Month string for time synthesis if 'time' coordinate is missing.
        date : int, optional
            Day for time synthesis if 'time' coordinate is missing.
        """
        self.ds = ds_model

        # Restore time synthesis logic if time is missing but metadata is provided
        if "time" not in self.ds.coords and all(v is not None for v in [dir_year, dir_month, date]):
            year = int(dir_year)  # type: ignore
            month_str = dir_month.replace(dir_year, "").replace("_", "")  # type: ignore
            month = int(month_str)
            start_date_model = datetime(year, month, date, 12, 0, 0)  # type: ignore
            time_coords = [start_date_model + timedelta(hours=x) for x in range(self.ds.sizes.get("time", 1))]
            self.ds = self.ds.assign_coords(time=time_coords)

    def get_itp(
        self,
        time_obs: Union[datetime, np.ndarray, xr.DataArray],
        lat_obs: Union[float, np.ndarray, xr.DataArray],
        lon_obs: Union[float, np.ndarray, xr.DataArray],
        variable_name: str,
    ) -> xr.DataArray:
        """
        Temporal + spatial interpolation for a given variable.

        🍃⚡ Aero Protocol: Backend-agnostic and preserves Dask laziness.
        Supports vectorized interpolation if inputs are arrays.
        Automatically handles various coordinate names (e.g., lat/latitude).

        Parameters
        ----------
        time_obs : datetime, np.ndarray, or xr.DataArray
            Observation time(s).
        lat_obs : float, np.ndarray, or xr.DataArray
            Observation latitude(s).
        lon_obs : float, np.ndarray, or xr.DataArray
            Observation longitude(s).
        variable_name : str
            The name of the variable to interpolate.

        Returns
        -------
        xr.DataArray
            Interpolated values. Returns NaN where points are outside the model domain.
        """
        # Map input names to dataset coordinate names
        interp_dict = {"time": time_obs}

        # Check for latitude/longitude names
        lat_name = next((name for name in ["lat", "latitude"] if name in self.ds.coords), "lat")
        lon_name = next((name for name in ["lon", "longitude"] if name in self.ds.coords), "lon")

        interp_dict[lat_name] = lat_obs
        interp_dict[lon_name] = lon_obs

        # xarray.Dataset.interp handles both scalar and vectorized interpolation.
        return self.ds[variable_name].interp(
            **interp_dict,
            method="linear",
        )
