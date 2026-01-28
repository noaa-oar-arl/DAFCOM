"""
0) import library
"""

import time

import numpy as np
import xarray as xr
from xregrid import Regridder

from src.forecast_mode.config import load_config
from src.forecast_mode.dataloader import Load_AOD, Load_Chem, Load_Meteo, Load_Observation, Load_Static_Data
from src.forecast_mode.utils import Regrid

"""
1) load config yaml file
"""
config = load_config()
start = time.time()
"""
2) load model data
"""
ufs_aod_folder = config["database"]["dynamic_inputs"]["aod"]
ufs_chem_folder = config["database"]["dynamic_inputs"]["surface_pollutants"]
ufs_meteo_folder = config["database"]["dynamic_inputs"]["meteorology"]
static_population_data = config["database"]["static_inputs"]["population"]
static_land_use_cover_data = config["database"]["static_inputs"]["land_use_cover"]
static_elevation_data = config["database"]["static_inputs"]["elevation"]


year_list = config["Time"]["year_list"]
month_list = config["Time"]["month_list"]
day_list = config["Time"]["day_list"]
OriSize_lat_ufs = config["Original_Size"]["model"]["dynamic_inputs"]["ufs"][0]
OriSize_lon_ufs = config["Original_Size"]["model"]["dynamic_inputs"]["ufs"][1]

AOD_dataset = Load_AOD.extract_data(ufs_aod_folder, year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_SP_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "surface_pressure", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
METEO_T2M_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "temperature_2m", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
METEO_RH_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "humidity", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
METEO_EVP_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "evaporation", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
METEO_WINDU_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "wind speed u", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
METEO_WINDV_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "wind speed v", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
METEO_PBLH_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "pblh", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
METEO_PREP_dataset = Load_Meteo.extract_data(
    ufs_meteo_folder, "precipitation", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
CHEM_PM25_dataset = Load_Chem.extract_data(
    ufs_chem_folder, "pm25", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs
)
CHEM_O3_dataset = Load_Chem.extract_data(ufs_chem_folder, "o3", year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
print("finish part 2")
print(f"Step 2 done, elapsed: {time.time() - start:.2f}s")

"""
3) regrid
"""
ur_lat = config["domain"]["upper_right_latitude"]
ll_lat = config["domain"]["lower_left_latitude"]
ur_lon = config["domain"]["upper_right_longitude"]
ll_lon = config["domain"]["lower_left_longitude"]
resolution = config["resolution"]
interpolation_method = config["interpolation_method"]

# 🍃⚡ Aero Protocol: Initialize a single Regridder to reuse weight matrices.
# This significantly speeds up processing when regridding multiple variables to the same grid.
lat_coords = np.arange(ur_lat, ll_lat, -resolution)
lon_coords = np.arange(ll_lon, ur_lon, resolution)
ds_out = xr.Dataset({"lat": (["lat"], lat_coords), "lon": (["lon"], lon_coords)})
regridder = Regridder(AOD_dataset, ds_out, method=interpolation_method)

final_aod = Regrid(AOD_dataset, "aod", regridder=regridder)
final_meteo_sp = Regrid(METEO_SP_dataset, "surface_pressure", regridder=regridder)
final_meteo_t2m = Regrid(METEO_T2M_dataset, "temperature_2m", regridder=regridder)
final_meteo_rh = Regrid(METEO_RH_dataset, "humidity", regridder=regridder)
final_meteo_evp = Regrid(METEO_EVP_dataset, "evaporation", regridder=regridder)
final_meteo_wind_u = Regrid(METEO_WINDU_dataset, "wind speed u", regridder=regridder)
final_meteo_wind_v = Regrid(METEO_WINDV_dataset, "wind speed v", regridder=regridder)
final_meteo_pblh = Regrid(METEO_PBLH_dataset, "pblh", regridder=regridder)
final_meteo_prep = Regrid(METEO_PREP_dataset, "precipitation", regridder=regridder)
final_chem_pm25 = Regrid(CHEM_PM25_dataset, "pm25", regridder=regridder)
final_chem_o3 = Regrid(CHEM_O3_dataset, "o3", regridder=regridder)
print("finish part 3")
print(f"Step 3 done, elapsed: {time.time() - start:.2f}s")

"""
4) static data no need regrid
"""
time_length = final_chem_pm25.dims["time"]

# 🍃⚡ Aero Protocol: Static data extraction now supports optional chunking.
final_static_elevation = Load_Static_Data.extract_elevation(
    static_elevation_data, "Elevation", time_length, chunks={"lat": 100, "lon": 100}
)
final_static_population = Load_Static_Data.extract_population(
    static_population_data, "Population", time_length, chunks={"lat": 100, "lon": 100}
)
final_static_land_use_cover = Load_Static_Data.extract_land_use_cover(
    static_land_use_cover_data, "luc", time_length, chunks={"lat": 100, "lon": 100}
)
print("finish part 4")
print(f"Step 4 done, elapsed: {time.time() - start:.2f}s")


"""
5) load observation from AirNow
"""
obs_airnow_folder = config["database"]["observations"]["airnow"]
# 🍃⚡ Aero Protocol: Observations are now returned as a lazy-friendly xarray Dataset.
# We use chunks={'site': 100} to enable parallel processing while maintaining a 2D structure.
final_obs_pm25 = Load_Observation.extract_airnow_pm25(
    obs_airnow_folder, "AirNow_20230801_20230831.nc", ll_lat, ur_lat, ll_lon, ur_lon, chunks={"site": 100}
)


print(f"finish part 5. Loaded {final_obs_pm25.pm25.count().values} valid observations.")
print(f"Step 5 done, elapsed: {time.time() - start:.2f}s")
