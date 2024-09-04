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

variable_name = 'tp'
'''
1) process data
'''
dir_1 = '/Volumes/Ext_Disk_2/Data/reanalysis_data/ERA-Interim_meteo_data/2023_US_CONUS/2023_08/'
f1 = xr.open_dataset(dir_1+'LandAndOcean_0p25.nc')

#this is old resolution variables
blh_old = f1[variable_name]

#get time index stay on UTC
base_time = dt(2023,8,1,0,0,0)
Time_series =[base_time + datetime.timedelta(hours = x) for x in range(len(blh_old))]

#get cdt time 2018-nov
blh_old = blh_old[:][:744]

def MeteoDaily(input_matrix):
    AAA = []
    aaa = np.zeros((97,241))
    for i in range(len(input_matrix)):
        if i% 24 == 23:
            aaa = aaa + input_matrix[i]
            aaa = aaa/24
            AAA.append(aaa)
            aaa = np.zeros((97,241))
            
        else:
            aaa = aaa + input_matrix[i]
            
    return AAA

blh_old_daily = MeteoDaily(blh_old)




#build new lat and lon

lat_new = [49- 0.25*X for X in range(97)]
lon_new = [-125+ 0.25*X for X in range(241)]







'''
2) write into netcdf file
'''

ft = Dataset(('step1_meteo_'+variable_name+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',97)
nlon = ft.createDimension('longitude',241)
nlayer = ft.createDimension('time',len(blh_old_daily))

lat_new_nc = ft.createVariable('latitude','f4',('latitude'))
lat_new_nc.units =''
lat_new_nc.description =''

lon_new_nc = ft.createVariable('longitude','f4',('longitude'))
lon_new_nc.units =''
lon_new_nc.description =''

variable_new_nc = ft.createVariable(variable_name,'f4',('time','latitude','longitude'))  
variable_new_nc .units =''                                               
variable_new_nc .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new_nc[:,:,:] = np.array(blh_old_daily)

ft.close()
























