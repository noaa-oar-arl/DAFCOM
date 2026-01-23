"""
Aero-compliant Data Loading Module.
🍃⚡ This module follows the Pangeo ecosystem best practices for lazy loading and backend-agnosticism.
"""

import fnmatch
import os
from datetime import datetime
from typing import List, Optional

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
    def _extract_static(filepath: str, variable_name: str, time_length: int, chunks: Optional[dict] = None) -> xr.DataArray:
        """Generic helper for static data extraction."""
        ds = xr.open_dataset(filepath, chunks=chunks or {})
        data = ds[variable_name]

        # Squeeze time if it exists
        if "time" in data.dims:
            data = data.isel(time=0, drop=True)

        # Expand along a new time dimension lazily using broadcasting
        # 🍃⚡ Aero Protocol: Vectorized expansion instead of Python lists.
        # We create a dummy time coordinate if none exists to facilitate broadcasting.
        dummy_time = pd.date_range("2000-01-01", periods=time_length, freq="h")
        data_expanded = data.expand_dims(time=dummy_time)

        return data_expanded

    @staticmethod
    def extract_elevation(
        static_elevation_data: str, variable_name: str, time_length: int, chunks: Optional[dict] = None
    ) -> xr.DataArray:
        """
        Extract elevation data and expand along time dimension.

        Parameters
        ----------
        static_elevation_data : str
            Path to the NetCDF file.
        variable_name : str
            Name of the variable to extract.
        time_length : int
            Number of time steps to replicate the static data for.
        chunks : dict, optional
            Dask chunk sizes for lazy loading.

        Returns
        -------
        xr.DataArray
            Static data expanded along the time dimension.
        """
        return Load_Static_Data._extract_static(static_elevation_data, variable_name, time_length, chunks=chunks)

    @staticmethod
    def extract_population(
        static_population_data: str, variable_name: str, time_length: int, chunks: Optional[dict] = None
    ) -> xr.DataArray:
        """
        Extract population data and expand along time dimension.

        Parameters
        ----------
        static_population_data : str
            Path to the NetCDF file.
        variable_name : str
            Name of the variable to extract.
        time_length : int
            Number of time steps to replicate the static data for.
        chunks : dict, optional
            Dask chunk sizes for lazy loading.

        Returns
        -------
        xr.DataArray
            Static data expanded along the time dimension.
        """
        return Load_Static_Data._extract_static(static_population_data, variable_name, time_length, chunks=chunks)

    @staticmethod
    def extract_land_use_cover(
        static_land_use_cover_data: str, variable_name: str, time_length: int, chunks: Optional[dict] = None
    ) -> xr.DataArray:
        """
        Extract land use cover data and expand along time dimension.

        Parameters
        ----------
        static_land_use_cover_data : str
            Path to the NetCDF file.
        variable_name : str
            Name of the variable to extract.
        time_length : int
            Number of time steps to replicate the static data for.
        chunks : dict, optional
            Dask chunk sizes for lazy loading.

        Returns
        -------
        xr.DataArray
            Static data expanded along the time dimension.
        """
        return Load_Static_Data._extract_static(static_land_use_cover_data, variable_name, time_length, chunks=chunks)


class Load_Observation:
    """Loader for AirNow Observation data."""

    @staticmethod
    def extract_airnow_pm25(
        obs_folder: str,
        obs_filename: str,
        ll_lat: float,
        ur_lat: float,
        ll_lon: float,
        ur_lon: float,
        chunks: Optional[dict] = None,
    ) -> xr.Dataset:
        """
        Extract AirNow PM2.5 data and return as a tabular xarray Dataset.

        🍃⚡ Aero Protocol: Backend-agnostic, supports Dask, and avoids explicit loops.

        Parameters
        ----------
        obs_folder : str
            Folder containing observation files.
        obs_filename : str
            Filename of the NetCDF observation file.
        ll_lat, ur_lat : float
            Latitude bounds.
        ll_lon, ur_lon : float
            Longitude bounds.
        chunks : dict, optional
            Dask chunk sizes for lazy loading.

        Returns
        -------
        xr.Dataset
            A tabular dataset of valid PM2.5 observations with coordinates.
        """
        filename = os.path.join(obs_folder, obs_filename)
        ds = xr.open_dataset(filename, chunks=chunks)

        # Basic cleanup
        pm25 = ds["PM2.5"]
        if "dim_1" in pm25.dims:  # Handle the [:, 0, :] from original code
            pm25 = pm25.isel(dim_1=0)

        # Hourly resampling - resample is lazy if data is dask-backed
        pm25_hr = pm25.resample(time="1h").mean()

        # Spatial filtering
        spatial_mask = (ds.latitude >= ll_lat) & (ds.latitude <= ur_lat) & (ds.longitude >= ll_lon) & (ds.longitude <= ur_lon)

        # Identify non-vacant sites (original logic was sum != -len(time))
        # Assuming -1 is the fill value from original code
        valid_sites = (pm25 != -1).any(dim="time") & spatial_mask

        ds_subset = ds.sel(site=valid_sites)
        pm25_subset_hr = pm25_hr.sel(site=valid_sites)

        # 🍃⚡ Aero Protocol: Avoid drop=True and forced stacking to maintain laziness.
        # The user can flatten the data (stack + dropna) when they are ready to compute.
        ds_final = pm25_subset_hr.where(pm25_subset_hr != -1).to_dataset(name="pm25")

        # Re-attach site-specific coordinates (latitude, longitude)
        ds_final["latitude"] = ds_subset.latitude
        ds_final["longitude"] = ds_subset.longitude

        # Round values for storage efficiency
        ds_final["pm25"] = ds_final["pm25"].round(4)

        ds_final.attrs["history"] = f"{datetime.now()}: Extracted AirNow PM2.5 via Aero Protocol."
        return ds_final
