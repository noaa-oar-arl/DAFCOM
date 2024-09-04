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


variable_name = 'v10'  #change here
'''
1) read inputs
'''
dir_1 = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/data/0_meteo/step1_fast/'
ds = xr.open_dataset(dir_1+'step1_meteo_'+variable_name+'.nc')

dr = ds[variable_name]
print(dr.shape)
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

ft = Dataset(('meteo_'+variable_name+'_0p01.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2400)
nlon = ft.createDimension('longitude',6000)
nlayer = ft.createDimension('time',len(dr))

lat_new_nc = ft.createVariable('lat','f4',('latitude'))
lat_new_nc.units =''
lat_new_nc.description =''

lon_new_nc = ft.createVariable('lon','f4',('longitude'))
lon_new_nc.units =''
lon_new_nc.description =''

variable_new_nc = ft.createVariable(variable_name,'f4',('time','latitude','longitude'))  
variable_new_nc .units =''                                               
variable_new_nc .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new_nc[:,:,:] = np.array(dr_out)

ft.close()


