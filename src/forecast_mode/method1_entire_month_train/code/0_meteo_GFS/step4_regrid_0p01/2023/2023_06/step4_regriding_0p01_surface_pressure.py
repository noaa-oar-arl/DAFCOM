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
import matplotlib.pyplot as plt
'''
1) read inputs
'''


#variable = 'evaporation'
# variable = 'pblh'
# variable = 'precipitation'
# variable = 'specific_humidity_2m'
variable = 'surface_pressure'
# variable = 't2m'
# variable = 'u10'
# variable = 'v10'


dir_1 = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/data/0_prepare_meteo_GFS/step3_daily_avr/2023/2023_06/'
ds = xr.open_dataset(dir_1+'step3_data_'+variable+'.nc')

dr = ds[variable][:]                                                           #change this this is by day in year 2019, 0 to 30 is Jan-2019
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

ft = Dataset(('GFS_meteo_0p01_'+variable+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2400)
nlon = ft.createDimension('longitude',6000)
nlayer = ft.createDimension('time',len(dr_out))

lat_new_nc = ft.createVariable('lat','f4',('latitude'))
lat_new_nc.units =''
lat_new_nc.description =''

lon_new_nc = ft.createVariable('lon','f4',('longitude'))
lon_new_nc.units =''
lon_new_nc.description =''

variable_new_nc = ft.createVariable(variable,'f4',('time','latitude','longitude'))  
variable_new_nc .units =''                                               
variable_new_nc .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new_nc[:,:,:] = np.array(dr_out)

ft.close()


























