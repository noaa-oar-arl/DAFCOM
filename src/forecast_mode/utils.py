#===========
#import library
#=============

from typing import List, Optional, Tuple, Dict, Any
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from netCDF4 import Dataset
# import xlsxwriter
from math import radians, sin, cos, asin, sqrt

#=============
# def class and functions
#==============
import xarray as xr
import xesmf as xe

def Regrid(ds, variable_name, ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method):
    #resolution type == float
    #interpolation_method type = str
    
    # Define output grid
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(ur_lat, ll_lat, -resolution)),
            "lon": (["lon"], np.arange(ll_lon, ur_lon, resolution))
        }
    )

    # Perform regridding
    regridder = xe.Regridder(ds, ds_out, interpolation_method)  #default interpolation_method = 'bilinear'
    dr_out = regridder(ds[variable_name])

    # Create final dataset with regridded data
    final_ds = xr.Dataset(
        {
            variable_name: (['time', 'latitude', 'longitude'], dr_out.values),
            'lat': ('latitude', np.arange(ur_lat, ll_lat, -resolution)),
            'lon': ('longitude', np.arange(ll_lon, ur_lon, resolution))
        },
        coords={
            'time': ds.time
        }
    )
    
    return final_ds





class Interpolator:

    def _find_grid_cell(self, lat_arr: np.ndarray, lon_arr: np.ndarray, lat_pt: float, lon_pt: float) -> Optional[Tuple[int,int,float,float]]:
        """Find indices i,j such that lat_arr[i] <= lat_pt <= lat_arr[i+1] (or reversed).
           Return (i, j, w_lat, w_lon) with weights in [0,1] relative to lower index.
           If outside grid return None.
        """
        lat_asc = np.all(np.diff(lat_arr) > 0)
        lon_asc = np.all(np.diff(lon_arr) > 0)
        if not lat_asc:
            lat_arr_proc = lat_arr[::-1]
            lat_index_reversed = True
        else:
            lat_arr_proc = lat_arr
            lat_index_reversed = False

        if not lon_asc:
            lon_arr_proc = lon_arr[::-1]
            lon_index_reversed = True
        else:
            lon_arr_proc = lon_arr
            lon_index_reversed = False

        # find insertion indices
        i = np.searchsorted(lat_arr_proc, lat_pt)
        j = np.searchsorted(lon_arr_proc, lon_pt)

        # need lower index
        if i == 0 or i >= len(lat_arr_proc):
            return None
        if j == 0 or j >= len(lon_arr_proc):
            return None

        i_low = i - 1
        j_low = j - 1

        # map back to original indices if reversed
        if lat_index_reversed:
            i_low = len(lat_arr) - 2 - i_low
            i_high = i_low + 1
        else:
            i_high = i_low + 1

        if lon_index_reversed:
            j_low = len(lon_arr) - 2 - j_low
            j_high = j_low + 1
        else:
            j_high = j_low + 1

        # compute weights in [0,1] relative to lower index
        lat_lo = lat_arr[i_low]
        lat_hi = lat_arr[i_high]
        lon_lo = lon_arr[j_low]
        lon_hi = lon_arr[j_high]
        # guard against zero division
        if lat_hi == lat_lo or lon_hi == lon_lo:
            return None
        w_lat = (lat_pt - lat_lo) / (lat_hi - lat_lo)
        w_lon = (lon_pt - lon_lo) / (lon_hi - lon_lo)
        return i_low, j_low, float(w_lat), float(w_lon)

    def _bilinear_interp(self, grid2d: np.ndarray, i: int, j: int, w_lat: float, w_lon: float) -> float:
        """Perform bilinear interpolation on 2D array grid2d using lower-left index (i,j) and weights."""
        a = (1.0 - w_lon) * grid2d[i, j] + w_lon * grid2d[i, j + 1]
        b = (1.0 - w_lon) * grid2d[i + 1, j] + w_lon * grid2d[i + 1, j + 1]
        val = (1.0 - w_lat) * a + w_lat * b
        return float(val)

    def get_itp(self, time_obs: datetime, lat_obs: float, lon_obs: float, variable_model: np.ndarray) -> Any:
        """Temporal + spatial interpolation for a given variable_model shaped (T, M, N) or (M,N)."""
        # temporal: find which hour slice to use
        # build model time list starting at local noon of date (same as earlier)
        year = int(self.dir_year)
        month = int(self.dir_month.replace(self.dir_year, '').replace('_', ''))
        start_date_model = datetime(year, month, self.date, 12, 0, 0)
        time_model_list = [start_date_model + timedelta(hours=x) for x in range(variable_model.shape[0])]

        # find time index (k) where model hour contains time_obs
        k = None
        for idx, t_start in enumerate(time_model_list):
            t_end = t_start + timedelta(hours=1)
            if t_start <= time_obs < t_end:
                k = idx
                break
        if k is None:
            return 'no_value'

        # spatial: variable_model[k] is 2D
        grid2d = variable_model[k] if variable_model.ndim == 3 else variable_model
        res = self._find_grid_cell(self.LAT, self.LON, lat_obs, lon_obs)
        if res is None:
            return 'no_value'
        i_low, j_low, w_lat, w_lon = res
        try:
            return self._bilinear_interp(grid2d, i_low, j_low, w_lat, w_lon)
        except Exception:
            return 'no_value'

