
from src.forecast_mode.config import load_config
from src.forecast_mode.utils import Regrid
from src.forecast_mode.dataloader import Load_AOD

'''
1) load config yaml file
'''
config =load_config()

'''
2) load model data
'''
ufs_aod_folder = config['database']['dynamic_inputs']['aod']
ufs_chem_folder =config['database']['dynamic_inputs']['surface_pollutants']
ufs_meteo_folder=config['database']['dynamic_inputs']['meteorology']


year_list = config['Time']['year_list']
month_list = config['Time']['month_list']
day_list = config['Time']['day_list']
OriSize_lat_aod = config['Original_Size']['model']['dynamic_inputs']['aod'][0]
OriSize_lon_aod = config['Original_Size']['model']['dynamic_inputs']['aod'][1]


AOD_dataset = Load_AOD.extract_data(ufs_aod_folder, year_list, month_list, day_list, OriSize_lat_aod, OriSize_lon_aod)

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










