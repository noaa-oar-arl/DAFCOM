'''
0) import library
'''
from src.forecast_mode.config import load_config
from src.forecast_mode.utils import Regrid
from src.forecast_mode.dataloader import Load_AOD
from src.forecast_mode.dataloader import Load_Meteo
from src.forecast_mode.dataloader import Load_Chem

import time

'''
1) load config yaml file
'''
config =load_config()
start = time.time()
'''
2) load model data
'''
ufs_aod_folder = config['database']['dynamic_inputs']['aod']
ufs_chem_folder =config['database']['dynamic_inputs']['surface_pollutants']
ufs_meteo_folder=config['database']['dynamic_inputs']['meteorology']


year_list = config['Time']['year_list']
month_list = config['Time']['month_list']
day_list = config['Time']['day_list']
OriSize_lat_ufs = config['Original_Size']['model']['dynamic_inputs']['ufs'][0]
OriSize_lon_ufs = config['Original_Size']['model']['dynamic_inputs']['ufs'][1]


AOD_dataset = Load_AOD.extract_data(ufs_aod_folder, year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_SP_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'surface_pressure', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_T2M_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'temperature_2m', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_RH_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'humidity', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_EVP_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'evaporation', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_WINDU_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'wind speed u', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_WINDV_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'wind speed v', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_PBLH_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'pblh', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
METEO_PREP_dataset = Load_Meteo.extract_data(ufs_meteo_folder, 'precipitation', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
CHEM_PM25_dataset = Load_Chem.extract_data(ufs_chem_folder, 'pm25', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
CHEM_O3_dataset = Load_Chem.extract_data(ufs_chem_folder, 'o3', year_list, month_list, day_list, OriSize_lat_ufs, OriSize_lon_ufs)
print('finish part 2')
print(f"Step 2 done, elapsed: {time.time() - start:.2f}s")

'''
3) regrid
'''
ur_lat = config['domain']['upper_right_latitude']
ll_lat = config['domain']['lower_left_latitude']
ur_lon = config['domain']['upper_right_longitude']
ll_lon = config['domain']['lower_left_longitude']
resolution = config['resolution']
interpolation_method = config['interpolation_method']




final_aod = Regrid(AOD_dataset, 'aod', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_sp = Regrid(METEO_SP_dataset, 'surface_pressure', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_t2m = Regrid(METEO_T2M_dataset, 'temperature_2m', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_rh = Regrid(METEO_RH_dataset, 'humidity', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_evp = Regrid(METEO_EVP_dataset, 'evaporation', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_wind_u = Regrid(METEO_WINDU_dataset, 'wind speed u', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_wind_v = Regrid(METEO_WINDV_dataset, 'wind speed v', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_pblh = Regrid(METEO_PBLH_dataset, 'pblh', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_meteo_prep = Regrid(METEO_PREP_dataset, 'precipitation', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_chem_pm25 = Regrid(CHEM_PM25_dataset, 'pm25', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
final_chem_o3 = Regrid(CHEM_O3_dataset, 'o3', ur_lat, ll_lat, ur_lon, ll_lon, resolution, interpolation_method)
print('finish part 3')
print(f"Step 3 done, elapsed: {time.time() - start:.2f}s")





