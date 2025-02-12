#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:07:08 2022

@author: btang1
"""

import numpy as np
from netCDF4 import Dataset

'''
0) initialization
'''

dir_ = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/data/'
dir_year = '2023'
dir_month = '2023_08'

dir_forecast = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/code/PM25_by_sample/2023/2023_08/'
'''
0) Meteorology
'''
print('0 meteo')
pattern_0 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_pblh.nc'
filename_0 = dir_ + pattern_0
f0 = Dataset(filename_0,'r')
LAT = f0.variables['lat'][:] #2400
LON = f0.variables['lon'][:] #6000

'''
1) This section is for UFS AQM Result as input
'''  
print('1 UFS AQM')  
pattern_1 = '5_prepare_UFS_AQM/step3_regrid_0p01/'+dir_year+'/'+dir_month+'/'+'UFS_0p01_regrid_data.nc'
filename_1 = dir_+ pattern_1
f1 = Dataset(filename_1,'r')

UFS_AQM_PM25_daily = f1.variables['pm25'][:] 

'''
2) This section is Forecast&Downscale Result (train 2023/07, predict 2023/08)
'''
print('2 Forecast&Downscale')
pattern_2 = 'forecast202308_train202306_pm25_0p01.nc'
filename_2 = dir_forecast+ pattern_2
f2 = Dataset(filename_2,'r')

Forecast_PM25_daily = f2.variables['pm25'][:] 

    
    
    
     
    
    
    
    
    
    






    
    
    
    
    
    
    
    
    
    
    
