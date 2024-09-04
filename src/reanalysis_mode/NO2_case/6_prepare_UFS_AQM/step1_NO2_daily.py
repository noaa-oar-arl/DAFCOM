#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:33:31 2022

@author: btang1
"""
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# # from pyhdf.SD import SD, SDC
# from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset

'''
1) import pm25 data
'''
dir_1  = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/data/6_UFS_AQM_NO2/'

pattern1= 'Surface_Map_no2.nc' #change 1

filename1 = dir_1+pattern1
f1 = Dataset(filename1,'r')   
v_list1 = list(f1.variables)  


CMAQ_PM25 = f1.variables['no2'][:]   #Stay on UTC  #change 2
lat = f1.variables['lat'][:]
lon = f1.variables['lon'][:]
 
# lat_real = np.zeros((265))
# for i in range(265):
#     lat_real[i] = lat[i][110]
# lon_real = lon[132]

print(np.shape(CMAQ_PM25))

lat_real = lat
lon_real = lon
'''
2) get daily averages
'''
        
def CMAQDaily(input_matrix):
    AAA = []
    aaa = np.zeros((488,775))
    for i in range(len(input_matrix)):
        if i <11:
            aaa = aaa + input_matrix[i]
        elif i == 11:
            aaa = aaa + input_matrix[i]
            aaa = aaa/12
            AAA.append(aaa)
            aaa = np.zeros((488,775))
        else:
            if (i-12)% 24 == 23:
                aaa = aaa + input_matrix[i]
                aaa = aaa/24
                AAA.append(aaa)
                aaa = np.zeros((488,775))
                
            else:
                aaa = aaa + input_matrix[i]
    #this if for last day 2023-08, 31 days
    aaa = aaa/23
    AAA.append(aaa)
    
            
    return AAA

CMAQ_pm25_daily = CMAQDaily(CMAQ_PM25)  
print(np.shape(CMAQ_pm25_daily))

'''
2)write in netcdf
'''
ft = Dataset((pattern1.replace('Surface_Map_no2','step1_data_NO2')),'w',format = 'NETCDF4')   #CHANGE 3
nlat = ft.createDimension('latitude',488)
nlon = ft.createDimension('longitude',775)
nlayer = ft.createDimension('time',len(CMAQ_pm25_daily))

lat_new = ft.createVariable('lat','f4',('latitude','longitude'))
lat_new.units =''
lat_new.description =''

lon_new = ft.createVariable('lon','f4',('latitude','longitude'))
lon_new.units =''
lon_new.description =''

variable_new = ft.createVariable('no2','f4',('time','latitude','longitude'))  #change 4
variable_new .units ='ppbv'                                  #change 5             
variable_new .description =''



lat_new[:] = np.array(lat_real)
lon_new[:] = np.array(lon_real)
variable_new[:,:,:] = np.array(CMAQ_pm25_daily)


ft.close()


f1.close()



