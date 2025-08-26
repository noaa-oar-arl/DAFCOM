'''
author: Beiming Tang
date: 04/29/2025
'''
import time
start_time = time.time()

import numpy as np
from netCDF4 import Dataset
import xarray as xr
import datetime
from datetime import datetime as dt
import pandas as pd
'''
0) initialize
'''
dir_source = '/Users/beiming_tang/Downloads/gs_data/'
pattern = 'AirNow_20250301_20250401.nc'
ur_lat = 49
ur_lon = -65
ll_lat = 25
ll_lon = -125
dir_output = '/Users/beiming_tang/Desktop/DAFCOM/src/forecast_mode/step2_obs_prepare/'

'''
1) load data from AirNow
'''
filename = dir_source+pattern
f1 = xr.open_dataset(filename)
lat = f1['latitude']       # 2273
lon = f1['longitude']      # 2273
time_ori = f1['time']      # 2017
pm25 = f1['PM2.5'][:,0,:]  #(2017,2273)
#o3 = f1['OZONE'][:,0,:]  
#no2 = f1['NO2'][:,0,:]  
#no = f1['NO'][:,0,:]  
#nox = f1['NOX'][:,0,:]  

print('step1',np.shape(pm25))

'''
2) Process data more efficiently using vectorized operations
'''
# Convert to pandas DataFrame for more efficient processing
pm25_avr_hr = pm25.resample(time='h').mean()
time_avr_hr = pd.to_datetime(time_ori.resample(time='h').mean().values)

print("Hourly averaged PM2.5 shape:", np.shape(pm25_avr_hr))

# Create masks for filtering
vacant_sites_mask = (pm25.sum(dim='time') != -len(time_ori))
lat_mask = (lat >= ll_lat) & (lat <= ur_lat)
lon_mask = (lon >= ll_lon) & (lon <= ur_lon)
valid_sites_mask = vacant_sites_mask & lat_mask & lon_mask

# Get indices of valid sites
valid_site_indices = np.where(valid_sites_mask)[0]

# Initialize lists with pre-calculated size for better memory efficiency
total_times = len(time_avr_hr)
valid_data = []

# Process data for valid sites
for site_idx in valid_site_indices:
    site_data = pm25_avr_hr[:, site_idx]
    valid_times_mask = site_data != -1
    
    if np.any(valid_times_mask):
        valid_times = time_avr_hr[valid_times_mask]
        valid_pm25 = np.round(site_data[valid_times_mask].values, 4)
        
        # Create data for this site
        site_data_dict = {
            'site_index': np.full_like(valid_pm25, site_idx),
            'time_utc': [str(t) for t in valid_times],
            'lat': np.full_like(valid_pm25, float(lat[site_idx])),
            'lon': np.full_like(valid_pm25, float(lon[site_idx])),
            'pm25': valid_pm25
        }
        valid_data.append(pd.DataFrame(site_data_dict))

# Combine all data
final_df = pd.concat(valid_data, ignore_index=True)

# Convert to final format
LOC_NUMBER_OBS_FINAL = final_df['site_index'].tolist()
TIME_OBS_FINAL = final_df['time_utc'].tolist()
PM_OBS_FINAL = final_df['pm25'].tolist()
LAT_OBS_FINAL = final_df['lat'].tolist()
LON_OBS_FINAL = final_df['lon'].tolist()


'''
3) Write directly to Excel using pandas
'''
# Save to Excel directly using pandas - more efficient than xlsxwriter for this case
output_filename = dir_output + pattern.replace('AirNow_', 'getobs_PM25_').replace('.nc', '.xlsx')
final_df.to_excel(output_filename, index=False, 
                 columns=['site_index', 'time_utc', 'lat', 'lon', 'pm25'],
                 header=['site_index', 'time_utc', 'lat', 'lon', 'airnow_pm25'])
print(f'Excel file saved to: {output_filename}')
print(f'Total processing time: {time.time() - start_time:.2f} seconds')




































