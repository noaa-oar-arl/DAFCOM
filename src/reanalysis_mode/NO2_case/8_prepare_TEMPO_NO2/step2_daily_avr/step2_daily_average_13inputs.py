#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:09:53 2022

@author: btang1
"""


import numpy as np
from netCDF4 import Dataset
import xarray as xr
# import xesmf as xe
# import matplotlib.pyplot as plt
import datetime 
from datetime import datetime as dt


'''
0) initializtion
'''

date_input ='0816'


'''
1) process data
'''
dir_1 = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/data/8_TEMPO_NO2/step1_extract/'
f1 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_0.nc')
f2 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_1.nc')
f3 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_2.nc')
f4 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_3.nc')
f5 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_4.nc')
f6 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_5.nc')
f7 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_6.nc')
f8 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_7.nc')
f9 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_8.nc')
f10 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_9.nc')
f11 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_10.nc')
f12 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_11.nc')
f13 = xr.open_dataset(dir_1+date_input+'/'+'step1_tempo_extract_'+date_input+'_12.nc')

lat_ori = f1['latitude']
lon_ori = f1['longitude']

'''
2) process using xrray
'''
no2_trop_1 = f1['vertical_no2_trop']
no2_trop_2 = f2['vertical_no2_trop']
no2_trop_3 = f3['vertical_no2_trop']
no2_trop_4 = f4['vertical_no2_trop']
no2_trop_5 = f5['vertical_no2_trop']
no2_trop_6 = f6['vertical_no2_trop']
no2_trop_7 = f7['vertical_no2_trop']
no2_trop_8 = f8['vertical_no2_trop']
no2_trop_9 = f9['vertical_no2_trop']
no2_trop_10 = f10['vertical_no2_trop']
no2_trop_11 = f11['vertical_no2_trop']
no2_trop_12 = f12['vertical_no2_trop']
no2_trop_13 = f13['vertical_no2_trop']

def FillZero(no2_trop_input):
    # print('before',no2_trop_1[0][0][0])
    no2_trop_input_masked = no2_trop_input.where(no2_trop_input != -1.e+30)
    # print('after',no2_trop_1_masked[0][0][0])
    no2_trop_input_fillnan = no2_trop_input_masked.fillna(0)
    # print('final',no2_trop_1_fillnan[0][0][0])  
    return no2_trop_input_fillnan

no2_trop_1_fillnan = FillZero(no2_trop_1)
no2_trop_2_fillnan = FillZero(no2_trop_2)
no2_trop_3_fillnan = FillZero(no2_trop_3)
no2_trop_4_fillnan = FillZero(no2_trop_4)
no2_trop_5_fillnan = FillZero(no2_trop_5)
no2_trop_6_fillnan = FillZero(no2_trop_6)
no2_trop_7_fillnan = FillZero(no2_trop_7)
no2_trop_8_fillnan = FillZero(no2_trop_8)
no2_trop_9_fillnan = FillZero(no2_trop_9)
no2_trop_10_fillnan = FillZero(no2_trop_10)
no2_trop_11_fillnan = FillZero(no2_trop_11)
no2_trop_12_fillnan = FillZero(no2_trop_12)
no2_trop_13_fillnan = FillZero(no2_trop_13)



avr_ori = (no2_trop_1_fillnan + no2_trop_2_fillnan+ no2_trop_3_fillnan +no2_trop_4_fillnan +no2_trop_5_fillnan +
           no2_trop_6_fillnan + no2_trop_7_fillnan+ no2_trop_8_fillnan +no2_trop_9_fillnan +no2_trop_10_fillnan +
           no2_trop_11_fillnan + no2_trop_12_fillnan + no2_trop_13_fillnan)/13                                             #change here

# avr_convert = avr_ori/(6.02*10**19)  #convert to TROPOMI unit
avr_convert = avr_ori




# build new lat and lon
# lat_new = lat_ori.reindex(latitude=list(reversed(lat_ori.latitude)))
lat_new  = lat_ori
lon_new = lon_ori
'''
2) write into netcdf file
'''

ft = Dataset(('step2_tempo_DailyAvr_'+date_input+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2950)
nlon = ft.createDimension('longitude',7750)
nlayer = ft.createDimension('time',1)

lat_new_nc = ft.createVariable('latitude','f4',('latitude'))
lat_new_nc.units =''
lat_new_nc.description =''

lon_new_nc = ft.createVariable('longitude','f4',('longitude'))
lon_new_nc.units =''
lon_new_nc.description =''

variable_new_nc = ft.createVariable('trop_no2','f4',('time','latitude','longitude'))  
variable_new_nc .units =''                                               
variable_new_nc .description =''

lat_new_nc[:] = lat_new
lon_new_nc[:] = lon_new
variable_new_nc[:,:,:] = avr_convert

ft.close()
























