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
dir_1 = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/data_CONUS_US/4_elevation/step1/'
input_list = ['1203','1204','1205','1206',
              '1303','1304','1305','1306','1307',
              '1403','1404','1405','1406','1407',
              '1503','1504','1505','1506','1507',
              '1603','1604','1605','1606','1607',
              '1703','1704','1705','1706','1707',
              '1803','1804','1805','1806','1807',
              '1903','1904','1905','1906','1907',
              '2003','2004','2005','2006','2007',
              '2103','2104','2105','2106','2107',
              '2203','2204','2205',
              '2303','2304']

lat_start_list =[50,45,40,35,
                 50,45,40,35,30,
                 50,45,40,35,30,
                 50,45,40,35,30,
                 50,45,40,35,30,
                 50,45,40,35,30,
                 50,45,40,35,30,
                 50,45,40,35,30,
                 50,45,40,35,30,
                 50,45,40,
                 50,45]

lat_end_list = [45,40,35,30,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,30,25,
                45,40,35,
                45,40]

lon_start_list = [-125,-125,-125,-125,  #12
                  -120,-120,-120,-120,-120,
                  -115,-115,-115,-115,-115,
                  -110,-110,-110,-110,-110,
                  -105,-105,-105,-105,-105,
                  -100,-100,-100,-100,-100,
                  -95,-95,-95,-95,-95,#18
                  -90,-90,-90,-90,-90,
                  -85,-85,-85,-85,-85,
                  -80,-80,-80,-80,-80,
                  -75,-75,-75,
                  -70,-70]

lon_end_list = [-120,-120,-120,-120, #12
                -115,-115,-115,-115,-115,
                -110,-110,-110,-110,-110,
                -105,-105,-105,-105,-105,
                -100,-100,-100,-100,-100,
                -95,-95,-95,-95,-95,
                -90,-90,-90,-90,-90,#18
                -85,-85,-85,-85,-85,
                -80,-80,-80,-80,-80,
                -75,-75,-75,-75,-75,
                -70,-70,-70,
                -65,-65]



for i in range(len(input_list)):
    ds = xr.open_dataset(dir_1+'Elevation_step1_'+input_list[i]+'.nc')
    
    dr = ds['Elevation']
    '''
    2) define output grid
    '''
    
    ds_out =xr.Dataset(
        {
            "lat": (["lat"], np.arange(lat_start_list[i], lat_end_list[i], -0.01), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(lon_start_list[i], lon_end_list[i], 0.01), {"units": "degrees_east"}),
        }
    )
    '''
    3) regrid 
    '''
    regridder = xe.Regridder(ds, ds_out, 'bilinear')
    dr_out  = regridder(dr,keep_attrs= True)
    
    
    lat_new = [lat_start_list[i]- 0.01*X for X in range(500)]
    lon_new = [lon_start_list[i]+ 0.01*X for X in range(500)]
    
    '''
    4) write into netcdf file
    '''
    
    ft = Dataset(('Elevation_0p01_regrid_'+input_list[i]+'.nc'),'w',format = 'NETCDF4')   
    nlat = ft.createDimension('latitude',500)
    nlon = ft.createDimension('longitude',500)
    nlayer = ft.createDimension('time',1)
    
    lat_new_nc = ft.createVariable('lat','f4',('latitude'))
    lat_new_nc.units =''
    lat_new_nc.description =''
    
    lon_new_nc = ft.createVariable('lon','f4',('longitude'))
    lon_new_nc.units =''
    lon_new_nc.description =''
    
    variable_new_nc = ft.createVariable('elevation','f4',('time','latitude','longitude'))  
    variable_new_nc .units =''                                               
    variable_new_nc .description =''
    
    lat_new_nc[:] = np.array(lat_new)
    lon_new_nc[:] = np.array(lon_new)
    variable_new_nc[:,:,:] = np.array(dr_out)
    
    ft.close()


