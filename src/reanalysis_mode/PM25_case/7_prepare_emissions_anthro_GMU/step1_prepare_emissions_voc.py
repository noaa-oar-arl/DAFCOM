#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:33:31 2022

@author: btang1
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# from pyhdf.SD import SD, SDC
from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset



name_variable_list = ['ACET','ACROLEIN','ALD2','ALDX','BENZ',
                      'BUTADIENE13','CH4','ETH','ETHA','ETHY',
                      'ETOH','FORM','IOLE','ISOP','KET',
                      'MEOH','NAPH','OLE','PAR','PRPA',
                      'FACD','TERP','TOL','XYLMN','AACD',
                      'APIN','SOAALK']


MW_list = [58.1, 56.1, 44, 58.1, 78.1,
           54, 16, 28, 30.1, 26, 
           46.1, 30, 56.1, 68.1, 72.1,
           32, 128.2, 42.1, 14, 44.1,
           46, 136.2, 92.1, 106.2, 60.1,
           136.2, 112]
'''
1) import emission data
'''
dir_1  = '/Volumes/Ext_Disk_2/Data/emission/Siqi_GMU_1km_anthro/2018/'
pattern_1= 'emis_mole_all_201711_US01_cmaq_cb6ae7_2017gb_17j_mean.nc'  
filename_1 = dir_1+pattern_1
f1 = Dataset(filename_1,'r')   

V_list1 = list(f1.variables)   

V_TOTAL = np.zeros((3177,5397))
for i in range(len(name_variable_list)):
    V1 = f1.variables[name_variable_list[i]][:][0][0] #(3177,5397)
    V_TOTAL =V_TOTAL +V1*MW_list[i]
    
print(np.max(V_TOTAL))


#IMPORT LAT AND LON INFO
dir_2 = '/Volumes/Ext_Disk_2/Data/emission/Siqi_GMU_1km_anthro/'
pattern_2 = 'GRID_LATLON_CENTRAL_US01.nc'
filename_2 = dir_2+pattern_2
f2 = Dataset(filename_2,'r')   

LAT = f2.variables['LAT'][:][0][0]  #(3177,5397)
LON = f2.variables['LON'][:][0][0]  #(3177,5397)  



'''
2) prepare
'''



Emission_Korea = V_TOTAL*3600*24/(1000)  #convert unit from g/s to kg/day


'''
2-3) slip lat directions
'''
LAT_list = LAT
LON_list = LON


Emission_Korea_nlayer = Emission_Korea.reshape((1,Emission_Korea.shape[0],Emission_Korea.shape[1]))

print(np.min(Emission_Korea_nlayer ))
print(np.max(Emission_Korea_nlayer))

'''
3)write in netcdf
'''
ft = Dataset('Emission_VOC_step1.nc','w',format = 'NETCDF4')                   #Change2
nlat = ft.createDimension('latitude',len(LAT_list))
nlon = ft.createDimension('longitude',len(LON_list[0]))
nlayer = ft.createDimension('time',1)

lat_new = ft.createVariable('lat','f4',('latitude','longitude'))
lat_new.units =''
lat_new.description =''

lon_new = ft.createVariable('lon','f4',('latitude','longitude'))
lon_new.units =''
lon_new.description =''

variable_new = ft.createVariable('E_voc','f4',('time','latitude','longitude')) #Change3
variable_new .units =''                                               
variable_new .description =''



lat_new[:] = np.array(LAT_list)
lon_new[:] = np.array(LON_list)
variable_new[:,:,:] = np.array(Emission_Korea_nlayer)


ft.close()
f1.close()



