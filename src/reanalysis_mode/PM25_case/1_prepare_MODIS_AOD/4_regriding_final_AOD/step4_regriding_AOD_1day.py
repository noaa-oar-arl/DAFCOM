#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:09:53 2022

@author: btang1
"""



import numpy as np
from netCDF4 import Dataset

import xarray as xr
import xesmf as xe
# import matplotlib.pyplot as plt



dateinput = '1101' #CHANGE ONLY HERE
'''
1) read inputs
'''
dir_1 = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/data_CONUS_US/1_MODIS_AOD/3_CAMS&KNN_AOD_filling/'
ds = xr.open_dataset(dir_1+'AOD_fill_'+dateinput+'.nc') 

dr = ds['AOD_fill']  
'''
2) define output grid
'''



ds_out =xr.Dataset(
    {
        "lat": (["lat"], np.arange(49, 25, -0.01), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(-125, -65, 0.01), {"units": "degrees_east"}),
    }
)
'''
3) regrid 
'''
regridder = xe.Regridder(ds, ds_out, 'bilinear')
dr_out  = regridder(dr,keep_attrs= True)


lat_new = [49- 0.01*X for X in range(2400)]
lon_new = [-125+ 0.01*X for X in range(6000)]

'''
4) write into netcdf file
'''

ft = Dataset(('final_AOD_regrid_0p01_'+dateinput+'.nc'),'w',format = 'NETCDF4')    
nlat = ft.createDimension('latitude',2400)
nlon = ft.createDimension('longitude',6000)
nlayer = ft.createDimension('time',1)

lat_new_nc = ft.createVariable('lat','f4',('latitude'))
lat_new_nc.units =''
lat_new_nc.description =''

lon_new_nc = ft.createVariable('lon','f4',('longitude'))
lon_new_nc.units =''
lon_new_nc.description =''

variable_new_nc = ft.createVariable('final_aod','f4',('time','latitude','longitude'))  
variable_new_nc .units =''                                               
variable_new_nc .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new_nc[:,:,:] = np.array(dr_out)

ft.close()


