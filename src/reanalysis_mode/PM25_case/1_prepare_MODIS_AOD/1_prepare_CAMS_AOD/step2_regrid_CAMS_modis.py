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
dir_1 = './'
ds = xr.open_dataset(dir_1+'daily_CAMS_AOD.nc') #change here

dr = ds['CAMS_AOD']  #change here
'''
2) define output grid
'''
aaa = np.abs((49-25)/2874)
bbb = np.abs((-125+65)/7184)


ds_out =xr.Dataset(
    {
        "lat": (["lat"], np.arange(49, 25+aaa, -aaa), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(-125, -65, bbb), {"units": "degrees_east"}),
    }
)
'''
3) regrid 
'''
regridder = xe.Regridder(ds, ds_out, 'bilinear')
dr_out  = regridder(dr,keep_attrs= True)


lat_new = [49- aaa*X for X in range(2874)]
lon_new = [-125+ bbb*X for X in range(7184)]

'''
4) write into netcdf file
'''

ft = Dataset(('CAMS_AOD_regrid.nc'),'w',format = 'NETCDF4')    #change here
nlat = ft.createDimension('latitude',2874)
nlon = ft.createDimension('longitude',7184)
nlayer = ft.createDimension('time',30)

lat_new_nc = ft.createVariable('lat','f4',('latitude'))
lat_new_nc.units =''
lat_new_nc.description =''

lon_new_nc = ft.createVariable('lon','f4',('longitude'))
lon_new_nc.units =''
lon_new_nc.description =''

variable_new_nc = ft.createVariable('cams_aod','f4',('time','latitude','longitude'))  #change here
variable_new_nc .units =''                                               
variable_new_nc .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new_nc[:,:,:] = np.array(dr_out)

ft.close()


