'''
author: Beiming Tang
date: 04/29/2025
'''
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import datetime
from datetime import datetime as dt
import pandas as pd

'''
1) load data from AirNow
'''
dir_airnow = '/data/aqf3/beiming.tang/DATA/gs_data/AirNow/2025/'
pattern = 'AirNow_20250401_20250501.nc'
filename = dir_airnow+pattern
f1 = xr.open_dataset(filename)
lat = f1['latitude']       # 2089
lon = f1['longitude']      # 2089
time_ori = f1['time']      # 2231
pm25 = f1['PM2.5'][:,0,:]  #(2231,2089)
#o3 = f1['OZONE'][:,0,:]  
#no2 = f1['NO2'][:,0,:]  
#no = f1['NO'][:,0,:]  
#nox = f1['NOX'][:,0,:]  

print(np.shape(pm25))

'''
2) find vacant site
'''
site_vacant_list = []
pm25_sum_by_site = pm25.sum(dim='time')
for i in range(len(pm25_sum_by_site)):
    if float(pm25_sum_by_site[i]) == -len(time_ori):
        site_vacant_list.append(i)

'''
3) average by hour
'''
pm25_avr_hr = pm25.resample(time = 'h').mean()     #(745,2231)
time_avr_hr = time_ori.resample(time= 'h').mean()  #2231

print(np.shape(pm25_avr_hr))

'''
4) get useful data
'''
pm25_new_list = []
time_new_list = []
lat_new_list = []
lon_new_list = []
index_new_list = []
for i in range(len(lat)): #number of sites ==2231
    if i not in site_vacant_list:
        if float(lat[i]) >= 25 and float(lat[i]) <= 49:
            if float(lon[i]) >= -125 and float(lon[i]) <= -65:
                for j in range(len(time_avr_hr)):  #time == 745
                #for j in range(30):
                    #print(i,j)
                    if float(pm25_avr_hr[j][i]) != -1:
                        pm25_new_list.append(np.round(float(pm25_avr_hr[j][i]),4))

                        time_here = time_avr_hr[j].values
                        time_formal = pd.to_datetime(time_here)
                        #time_new_list.append(time_formal.to_pydatetime())
                        time_new_list.append(str(time_formal))
                       
                        lat_new_list.append(float(lat[i]))
                        lon_new_list.append(float(lon[i]))
                        index_new_list.append(i)


'''
5) output
'''
LOC_NUMBER_OBS_FINAL = index_new_list
TIME_OBS_FINAL = time_new_list
PM_OBS_FINAL = pm25_new_list
LAT_OBS_FINAL = lat_new_list
LON_OBS_FINAL = lon_new_list


'''
6) write into xlsx excel
'''
import xlsxwriter
my_list = [LOC_NUMBER_OBS_FINAL,TIME_OBS_FINAL,LAT_OBS_FINAL,LON_OBS_FINAL,PM_OBS_FINAL]
workbook = xlsxwriter.Workbook(pattern.replace('AirNow_','getobs_PM25_').replace('.nc','.xlsx'))
worksheet = workbook.add_worksheet()
worksheet.write(0,0,'site_index')
worksheet.write(0,1,'time_utc')
worksheet.write(0,2,'lat')
worksheet.write(0,3,'lon')
worksheet.write(0,4,'airnow_pm25')
for row_num, row_data in enumerate(my_list):
    for col_num, data in enumerate(row_data):
        worksheet.write(col_num+1, row_num, data)
workbook.close()






































