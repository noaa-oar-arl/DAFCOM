#author:beiming tang
#date: 04/23/2025

import numpy as np
import os
import sys
from  netCDF4 import Dataset
import fnmatch

'''
1) get data in
'''
variable_name_list = ['surface_pressure','t2m','specific_humidity_2m','evaporation',
                      'u10','v10','pblh','precipitation']
month='04'

dir_data = '/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Code/step1_input_prepare/0_meteo/step1_bev/2026/'+month+'/'
pattern  = 'BEV_meteo*'

data_list_raw = fnmatch.filter(os.listdir(dir_data),pattern)
filelist = np.sort(data_list_raw)

for variable_name in variable_name_list:
    ft =  Dataset(('Surface_Map_'+month+'_'+variable_name+'.nc'),'w',format = 'NETCDF4')
    time = ft.createDimension('time',len(filelist))
    lat  = ft.createDimension('lat',488)
    lon  = ft.createDimension('lon',775)

    time_new = ft.createVariable('time','S1',('time',))
    time_new.units = ''
    time_new.description = ''

    lat = ft.createVariable('lat','f4',('lat','lon'))
    lat.units=''
    lat.description = ''

    lon = ft.createVariable('lon','f4',('lat','lon'))
    lon.units=''
    lon.description = ''

    variable = ft.createVariable(variable_name,'f4',('time','lat','lon'))
    variable.units = ''
    variable.description =  ''

    for i in range(len(filelist)):

        filename = dir_data+filelist[i]
        f = Dataset(filename,'r')

        time_w = filelist[i].replace('BEV_meteo_','')
        lat_w  = f.variables['lat'][:]
        lon_w  = f.variables['lon'][:]
        variable_w = f.variables[variable_name][0,:]
        
        time_new[:] = time_w
        lat[:,:] = lat_w
        lon[:,:] = lon_w
        variable[i,:,:]=variable_w

        f.close()
    ft.close()
































