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

date_input ='0827'


'''
1) process data
'''
dir_1 = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/data/8_TROPOMI_NO2/step2_final_regrid_0p01/'
f1 = xr.open_dataset(dir_1+'final_NO2_TROP_regrid_0p01_'+date_input+'.nc')


lat_ori = f1['lat']
lon_ori = f1['lon']

'''
2) process using xrray
'''
no2_tropomi_ori = f1['final_no2_trop']
# print(no2_tropomi_ori[0][1000][2000])

no2_tropomi_convert = no2_tropomi_ori *(6.02*10**19)
# print(no2_tropomi_convert[0][1000][2000])


# build new lat and lon
# lat_new = lat_ori.reindex(latitude=list(reversed(lat_ori.latitude)))
lat_new  = lat_ori
lon_new = lon_ori

'''
2) write into netcdf file
'''

ft = Dataset(('tropomi_no2_trop_regrid_0p01_'+date_input+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2400)
nlon = ft.createDimension('longitude',6000)
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
variable_new_nc[:,:,:] = no2_tropomi_convert

ft.close()
























