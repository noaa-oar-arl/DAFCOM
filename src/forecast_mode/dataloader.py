"""
Aero-compliant Data Loading Module.
🍃⚡ This module follows the Pangeo ecosystem best practices for lazy loading and backend-agnosticism.
"""

import fnmatch
import os
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import xarray as xr


def _get_file_paths(folder: str, year_list: List[str], month_list: List[str], day_list: List[str], pattern: str) -> List[str]:
    """
    Helper to discover file paths based on nested directory structure.

    Parameters
    ----------
    folder : str
        Base directory.
    year_list : List[str]
        Years to include.
    month_list : List[str]
        Months to include.
    day_list : List[str]
        Days to include.
    pattern : str
        Glob-style pattern for filenames.

    Returns
    -------
    List[str]
        Sorted list of absolute file paths.
    """
    file_paths = []
    for year in year_list:
        for month in month_list:
            for day in day_list:
                dir_data = os.path.join(folder, str(year), f"{year}{month}", f"{year}{month}{day}")
                if os.path.exists(dir_data):
                    files = sorted(fnmatch.filter(os.listdir(dir_data), pattern))
                    file_paths.extend([os.path.join(dir_data, f) for f in files])
    return file_paths


class Load_AOD:
    """Loader for Aerosol Optical Depth (AOD) data."""

    @staticmethod
    def extract_data(
        aod_folder: str,
        year_list: List[str],
        month_list: List[str],
        day_list: List[str],
        size_lat_ufs: int,
        size_lon_ufs: int,
        chunks: Optional[dict] = None,
    ) -> xr.Dataset:
        """
        Extract AOD data and return as an xarray Dataset.

        Parameters
        ----------
        aod_folder : str
            Base folder for AOD data.
        year_list : List[str]
            Years to process.
        month_list : List[str]
            Months to process.
        day_list : List[str]
            Days to process.
        size_lat_ufs : int
            Original latitude size (for API compatibility).
        size_lon_ufs : int
            Original longitude size (for API compatibility).
        chunks : dict, optional
            Dask chunk sizes for lazy loading.

        Returns
        -------
        xr.Dataset
            The combined AOD dataset.
        """
        file_paths = _get_file_paths(aod_folder, year_list, month_list, day_list, "aqm.t12z.phy.f*")
        if not file_paths:
            raise FileNotFoundError(f"No AOD files found in {aod_folder}")

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            # Handle potential extra dimensions and fix longitude
            if "aod" in ds.data_vars and ds.aod.ndim > 3:
                # Squeeze the first non-spatial/non-temporal dimension if it's size 1
                ds = ds.isel({ds.aod.dims[1]: 0})
            if "lon" in ds.coords:
                ds = ds.assign_coords(lon=ds.lon - 360)
            elif "lon" in ds.data_vars:
                ds = ds.assign(lon=ds.lon - 360)
            return ds

        ds = xr.open_mfdataset(
            file_paths,
            chunks=chunks,
            preprocess=preprocess,
            combine="nested",
            concat_dim="time",
            coords="minimal",
            compat="override",
        )
        ds.attrs["history"] = f"{datetime.now()}: Extracted AOD data via Aero Protocol."
        return ds


class Load_Meteo:
    """Loader for Meteorology data."""

    VAR_MAPPING = {
        "surface_pressure": "pressfc",
        "temperature_2m": "tmp2m",
        "humidity": "spfh2m",
        "evaporation": ["evbs_ave", "evcw_ave"],
        "wind speed u": "ugrd10m",
        "wind speed v": "vgrd10m",
        "pblh": "hpbl",
        "precipitation": "tprcp",
    }

    @staticmethod
    def extract_data(
        meteo_folder: str,
        variable_name: str,
        year_list: List[str],
        month_list: List[str],
        day_list: List[str],
        size_lat_ufs: int,
        size_lon_ufs: int,
        chunks: Optional[dict] = None,
    ) -> xr.Dataset:
        """
        Extract Meteorology data and return as an xarray Dataset.

        Parameters
        ----------
        meteo_folder : str
            Base folder for Meteorology data.
        variable_name : str
            The name of the variable to extract.
        year_list : List[str]
            Years to process.
        month_list : List[str]
            Months to process.
        day_list : List[str]
            Days to process.
        size_lat_ufs : int
            Original latitude size (for API compatibility).
        size_lon_ufs : int
            Original longitude size (for API compatibility).
        chunks : dict, optional
            Dask chunk sizes for lazy loading.

        Returns
        -------
        xr.Dataset
            The combined Meteorology dataset.
        """
        file_paths = _get_file_paths(meteo_folder, year_list, month_list, day_list, "aqm.t12z.phy.f*")
        if not file_paths:
            raise FileNotFoundError(f"No Meteo files found in {meteo_folder}")

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            target_var = Load_Meteo.VAR_MAPPING.get(variable_name)
            if target_var is None:
                raise ValueError(f"Unknown variable: {variable_name}")

            if isinstance(target_var, list):
                # Sum components (e.g., evaporation)
                data = sum(ds[v] for v in target_var)
                ds = ds.assign({variable_name: data})
            else:
                ds = ds.rename({target_var: variable_name})

            # Handle extra dimensions and fix longitude
            if ds[variable_name].ndim > 3:
                ds = ds.isel({ds[variable_name].dims[1]: 0})
            if "lon" in ds.coords:
                ds = ds.assign_coords(lon=ds.lon - 360)
            return ds[[variable_name, "lat", "lon"]]

        ds = xr.open_mfdataset(
            file_paths,
            chunks=chunks,
            preprocess=preprocess,
            combine="nested",
            concat_dim="time",
            coords="minimal",
            compat="override",
        )
        ds.attrs["history"] = f"{datetime.now()}: Extracted {variable_name} via Aero Protocol."
        return ds


class Load_Chem:
    """Loader for Chemistry data."""

    VAR_MAPPING = {"pm25": "PM25_TOT", "o3": "o3"}

    @staticmethod
    def extract_data(
        chem_folder: str,
        variable_name: str,
        year_list: List[str],
        month_list: List[str],
        day_list: List[str],
        size_lat_ufs: int,
        size_lon_ufs: int,
        chunks: Optional[dict] = None,
    ) -> xr.Dataset:
        """
        Extract Chemistry data and return as an xarray Dataset.

        Parameters
        ----------
        chem_folder : str
            Base folder for Chemistry data.
        variable_name : str
            The name of the variable to extract.
        year_list : List[str]
            Years to process.
        month_list : List[str]
            Months to process.
        day_list : List[str]
            Days to process.
        size_lat_ufs : int
            Original latitude size (for API compatibility).
        size_lon_ufs : int
            Original longitude size (for API compatibility).
        chunks : dict, optional
            Dask chunk sizes for lazy loading.

        Returns
        -------
        xr.Dataset
            The combined Chemistry dataset.
        """
        file_paths = _get_file_paths(chem_folder, year_list, month_list, day_list, "aqm.t12z.chem_sfc.nc")
        if not file_paths:
            raise FileNotFoundError(f"No Chemistry files found in {chem_folder}")

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            target_var = Load_Chem.VAR_MAPPING.get(variable_name)
            if target_var is None:
                raise ValueError(f"Unknown variable: {variable_name}")

            ds = ds.rename({target_var: variable_name})
            # Squeeze second dimension if it exists (e.g., level)
            if ds[variable_name].ndim == 4:
                ds = ds.isel({ds[variable_name].dims[1]: 0})
            if "lon" in ds.coords:
                ds = ds.assign_coords(lon=ds.lon - 360)
            return ds[[variable_name, "lat", "lon"]]

        # Using combine='by_coords' as chemistry files usually have time coordinates
        ds = xr.open_mfdataset(
            file_paths,
            chunks=chunks,
            preprocess=preprocess,
            combine="by_coords",
            coords="minimal",
            compat="override",
        )
        ds.attrs["history"] = f"{datetime.now()}: Extracted {variable_name} via Aero Protocol."
        return ds


class Load_Static_Data:
    """Loader for Static data (Elevation, Population, Land Use)."""

    @staticmethod
    def extract_elevation(static_elevation_data: str, variable_name: str, time_length: int) -> List[xr.DataArray]:
        """
        Extract elevation data.

        Returns a list of DataArrays to maintain API compatibility while preserving laziness.

        Parameters
        ----------
        static_elevation_data : str
            Path to the NetCDF file.
        variable_name : str
            Name of the variable to extract.
        time_length : int
            Number of time steps to replicate the static data for.

        Returns
        -------
        List[xr.DataArray]
            List of static data arrays.
        """
        ds = xr.open_dataset(static_elevation_data, chunks={})
        data = ds[variable_name].isel(time=0) if "time" in ds.dims else ds[variable_name]
        return [data] * time_length

    @staticmethod
    def extract_population(static_population_data: str, variable_name: str, time_length: int) -> List[xr.DataArray]:
        """Extract population data."""
        ds = xr.open_dataset(static_population_data, chunks={})
        data = ds[variable_name].isel(time=0) if "time" in ds.dims else ds[variable_name]
        return [data] * time_length

    @staticmethod
    def extract_land_use_cover(static_land_use_cover_data: str, variable_name: str, time_length: int) -> List[xr.DataArray]:
        """Extract land use cover data."""
        ds = xr.open_dataset(static_land_use_cover_data, chunks={})
        data = ds[variable_name].isel(time=0) if "time" in ds.dims else ds[variable_name]
        return [data] * time_length


class Load_Observation:
    """Loader for AirNow Observation data."""

    @staticmethod
    def extract_airnow_pm25(
        obs_folder: str, obs_filename: str, ll_lat: float, ur_lat: float, ll_lon: float, ur_lon: float
    ) -> Tuple[List[int], List[str], List[float], List[float], List[float]]:
        """
        Extract AirNow PM2.5 data and return as lists.
        🍃⚡ Refactored for efficiency using xarray vectorization.
        """
        filename = os.path.join(obs_folder, obs_filename)
        ds = xr.open_dataset(filename)

        # Basic cleanup
        pm25 = ds["PM2.5"]
        if "dim_1" in pm25.dims:  # Handle the [:, 0, :] from original code
            pm25 = pm25.isel(dim_1=0)

        # Hourly resampling
        pm25_hr = pm25.resample(time="1h").mean()

        # Spatial filtering
        mask = (ds.latitude >= ll_lat) & (ds.latitude <= ur_lat) & (ds.longitude >= ll_lon) & (ds.longitude <= ur_lon)

        # Identify non-vacant sites (original logic was sum != -len(time))
        # Assuming -1 is the fill value from original code
        valid_sites = (pm25 != -1).any(dim="time") & mask

        ds_valid = ds.sel(site=valid_sites)
        pm25_valid_hr = pm25_hr.sel(site=valid_sites)

        # Convert to the expected list format (API compatibility)
        index_new_list = []
        time_new_list = []
        pm25_new_list = []
        lat_new_list = []
        lon_new_list = []

        # We still need a loop to flatten to the specific format requested,
        # but it's now only over valid data.
        # This is still a "Lazy Breaker" due to the return type, but significantly faster.
        for i in range(len(ds_valid.site)):
            site_data = pm25_valid_hr.isel(site=i)
            site_lat = float(ds_valid.latitude.isel(site=i))
            site_lon = float(ds_valid.longitude.isel(site=i))
            site_idx = int(ds_valid.site.isel(site=i))  # Assuming site is a coord with indices

            # Find non-missing times
            valid_times = site_data.where(site_data != -1, drop=True)
            for t in range(len(valid_times.time)):
                val = float(valid_times.isel(time=t))
                t_val = pd.to_datetime(valid_times.time.isel(time=t).values)

                index_new_list.append(site_idx)
                time_new_list.append(str(t_val))
                pm25_new_list.append(round(val, 4))
                lat_new_list.append(site_lat)
                lon_new_list.append(site_lon)

        return index_new_list, time_new_list, pm25_new_list, lat_new_list, lon_new_list
