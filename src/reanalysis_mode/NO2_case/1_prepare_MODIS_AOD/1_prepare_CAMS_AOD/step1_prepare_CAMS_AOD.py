#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:07:08 2022

@author: btang1
"""

import numpy as np
# import os
# from scipy import spatial
# import fnmatch
from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.basemap import Basemap
# import codecs
# import datetime
# from datetime import datetime as dt
# import h5py

'''
1) read in CAMS AOD and make daily
'''    
dir_ = '/Volumes/Ext_Disk_2/Data/reanalysis_data/CAMS/Over_CONUS_US/2023/'
pattern = 'levtype_sfc_AOD550_2023Aug_CONUS.nc'

filename = dir_+pattern
f = Dataset(filename,'r')

CAMS_AOD = f.variables['aod550'][:]
LAT_AOD = f.variables['latitude']   
LON_AOD = f.variables['longitude']  

def CAMSDaily(input_matrix):
    AAA = []
    aaa = np.zeros((33,81))
    for i in range(len(input_matrix)):
        if i% 8 == 7:
            aaa = aaa + input_matrix[i]
            aaa = aaa/8
            AAA.append(aaa)
            aaa = np.zeros((33,81))
            
        else:
            aaa = aaa + input_matrix[i]
            
    return AAA


CAMS_aod_daily = CAMSDaily(CAMS_AOD)  
  



'''
2) write into netcdf file
'''

ft = Dataset(('daily_CAMS_AOD.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',33)
nlon = ft.createDimension('longitude',81)
nlayer = ft.createDimension('time',len(CAMS_aod_daily))

lat_new_nc = ft.createVariable('latitude','f4',('latitude'))
lat_new_nc.units =''
lat_new_nc.description =''

lon_new_nc = ft.createVariable('longitude','f4',('longitude'))
lon_new_nc.units =''
lon_new_nc.description =''

variable_new_nc = ft.createVariable('CAMS_AOD','f4',('time','latitude','longitude'))  
variable_new_nc .units =''                                               
variable_new_nc .description =''

lat_new_nc[:] = np.array(LAT_AOD)
lon_new_nc[:] = np.array(LON_AOD)
variable_new_nc[:,:,:] = np.array(CAMS_aod_daily)

ft.close()






















    
    
    
    
    
    
    
    
    
    
    