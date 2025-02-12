#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:33:31 2022

@author: btang1
"""
import numpy as np
from netCDF4 import Dataset

'''
1) import pm25 data
'''

varibale_name_list=['AOD']

for varibale in varibale_name_list:
    dir_1  = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/data/1_prepare_aod_gefs/step2_surface_map/2023/2023_06/'
    pattern1= 'Surface_Map_'+varibale+'.nc'
    
    filename1 = dir_1+pattern1
    f1 = Dataset(filename1,'r')   
    v_list1 = list(f1.variables)  
    
    
    CMAQ_PM25 = f1.variables[varibale][:]   #Stay on UTC
    lat = f1.variables['lat'][:]
    lon = f1.variables['lon'][:]
     
    
    lat_real = lat
    lon_real = lon
    '''
    2) get daily averages
    '''
            
    def CMAQDaily(input_matrix):
        AAA = []
        aaa = np.zeros((721,1440))
        for i in range(len(input_matrix)):
            if i% 8 == 7:
                aaa = aaa + input_matrix[i]
                aaa = aaa/8
                AAA.append(aaa)
                aaa = np.zeros((721,1440))
                
            else:
                aaa = aaa + input_matrix[i]
                
        return AAA
    
    CMAQ_pm25_daily = CMAQDaily(CMAQ_PM25)  
    
    '''
    2)write in netcdf
    '''
    ft = Dataset((pattern1.replace('Surface_Map_'+varibale,'step3_data_'+varibale)),'w',format = 'NETCDF4')   
    nlat = ft.createDimension('latitude',721)
    nlon = ft.createDimension('longitude',1440)
    nlayer = ft.createDimension('time',len(CMAQ_pm25_daily))
    
    lat_new = ft.createVariable('lat','f4',('latitude','longitude'))
    lat_new.units =''
    lat_new.description =''
    
    lon_new = ft.createVariable('lon','f4',('latitude','longitude'))
    lon_new.units =''
    lon_new.description =''
    
    variable_new = ft.createVariable(varibale,'f4',('time','latitude','longitude'))  
    variable_new .units =''                                               
    variable_new .description =''
    
    
    
    lat_new[:] = np.array(lat_real)
    lon_new[:] = np.array(lon_real)
    variable_new[:,:,:] = np.array(CMAQ_pm25_daily)
    
    
    ft.close()
    
    
    f1.close()



