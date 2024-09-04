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
import datetime 
from datetime import datetime as dt
import math

variable_name = 'v10'

'''
1) process 0p10 land_only data
'''
dir_1 = '/Volumes/Ext_Disk_2/Data/reanalysis_data/ERA-Interim_meteo_data/2018_US_CONUS/'
f1 = xr.open_dataset(dir_1+'land_only_0p10_2018Nov_CONUS.nc')
f2 = xr.open_dataset(dir_1+'land_only_0p10_2018Dec1st_CONUS.nc')

#combine land-only in time (nov + dec 1st)
variable_land_1 = f1[variable_name]
variable_land_2 = f2[variable_name]

variable_land_combine = np.concatenate((variable_land_1, variable_land_2), axis = 0)


#stay time 2018-nov
variable_land_combine = variable_land_combine[:][:720] #shape is (720,251,601)

'''
2) process 0p25 land and ocecan data
'''
dir_3 = '/Volumes/Ext_Disk_2/Data/reanalysis_data/ERA-Interim_meteo_data/2018_US_CONUS/'
f3 = xr.open_dataset(dir_1+'land_and_ocean_0p25_2018Nov&Dec_CONUS.nc')


variable_ocean = f3[variable_name]
#get cdt time 2018-nov
variable_ocean = variable_ocean[:][:720] #shape is (720,101,141)

#re-grid ocean to land resolution
ds_out =xr.Dataset(
    {
        "lat": (["lat"], np.arange(50, 25-0.10, -0.10), {"units": "degrees_north"}),
        "lon": (["lon"], np.arange(-125, -65+0.10, 0.10), {"units": "degrees_east"}),
    }
)

regridder = xe.Regridder(f3 , ds_out, 'bilinear')
dr_out  = regridder(variable_ocean ,keep_attrs= True)


# lat_new = [50- 0.10*X for X in range(251)]
# lon_new = [-125+ 0.10*X for X in range(351)]

'''
3) fill land-only with ocean
'''
variable_final = np.zeros((720,251,601))
for i in range(len(variable_land_combine)):  #this is time 720
    print(i)
    for j in range(len(variable_land_combine[0])):   #this is lat 251
        for k in range(len(variable_land_combine[0][0])):  #this is lon 351
 
            if math.isnan(variable_land_combine[i][j][k]) == True:
                variable_final[i][j][k] = dr_out[i][j][k]
                
            else:
                variable_final[i][j][k] = variable_land_combine[i][j][k]
                



'''
4) get daily averages
'''


def MeteoDaily(input_matrix):
    AAA = []
    aaa = np.zeros((251,601))
    for i in range(len(input_matrix)):
        if i% 24 == 23:
            aaa = aaa + input_matrix[i]
            aaa = aaa/24
            AAA.append(aaa)
            aaa = np.zeros((251,601))
            
        else:
            aaa = aaa + input_matrix[i]
            
    return AAA

variable_final_daily = MeteoDaily(variable_final )




#build new lat and lon

lat_new = [50- 0.10*X for X in range(251)]
lon_new = [-125+ 0.10*X for X in range(601)]


'''
5) write into netcdf file
'''

ft = Dataset(('step1_meteo_'+variable_name+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',251)
nlon = ft.createDimension('longitude',601)
nlayer = ft.createDimension('time',len(variable_final_daily))

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
variable_new_nc[:,:,:] = np.array(variable_final_daily)

ft.close()
























