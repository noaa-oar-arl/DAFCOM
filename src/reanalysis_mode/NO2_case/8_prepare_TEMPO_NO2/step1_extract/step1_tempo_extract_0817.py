#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:09:53 2022

@author: btang1
"""


import numpy as np
from netCDF4 import Dataset
import xarray as xr
# import xesmf as xe
# import matplotlib.pyplot as plt
import datetime 
from datetime import datetime as dt

'''
0) initialization
'''
date_input = '0817'

pattern_1 = 'TEMPO_NO2_L3_V03_20230817T163915Z_S001.nc'
pattern_2 = 'TEMPO_NO2_L3_V03_20230817T180915Z_S002.nc'
pattern_3 = 'TEMPO_NO2_L3_V03_20230817T195633Z_S003.nc'
pattern_4 = 'TEMPO_NO2_L3_V03_20230817T210845Z_S004.nc'
pattern_5 = 'TEMPO_NO2_L3_V03_20230817T223806Z_S005.nc'
pattern_6 = 'TEMPO_NO2_L3_V03_20230817T234037Z_S006.nc'


pattern_list = []
pattern_list.append(pattern_1)
pattern_list.append(pattern_2)
pattern_list.append(pattern_3)
pattern_list.append(pattern_4)
pattern_list.append(pattern_5)
pattern_list.append(pattern_6)


'''
1) oepn data
'''
for i in range(len(pattern_list)):
    
    dir_1 = '/Volumes/Ext_Disk_2/Data/satellite_data/TEMPO_NO2/'+date_input+'/'
    f1 = xr.open_dataset(dir_1+pattern_list[i])
    gp1 = Dataset(dir_1+pattern_list[i])
    
    latitude_ori = f1['latitude']   #2950
    longitude_ori = f1['longitude'] #7750
    
    print(str(i)+' lat=',len(latitude_ori ),' lon=',len(longitude_ori))
    # print(str(i)+' lon=',len(longitude_ori))
    
    product = gp1['/product/']
    trop_no2 = product['vertical_column_troposphere']  #unit = molecules/cm^2, fill_value = -1e+30
    
    trop_no2_unit = trop_no2.units
    trop_no2_nanvalue = trop_no2._FillValue
    
    
    # print(trop_no2[0][0][1000])
    qa = product['main_data_quality_flag']       #value = [0,1,2], fill_value = -9999
    qa_nanvalue = qa._FillValue
    

    
    
    
    
    # #build new lat and lon
    lat_new = latitude_ori
    lon_new = longitude_ori
    
    '''
    2) write into netcdf file
    '''
    
    ft = Dataset(('step1_tempo_extract_'+date_input+'_'+str(i)+'.nc'),'w',format = 'NETCDF4')   
    nlat = ft.createDimension('latitude',len(lat_new))
    nlon = ft.createDimension('longitude',len(lon_new))
    nlayer = ft.createDimension('time',1)
    
    lat_new_nc = ft.createVariable('latitude','f4',('latitude'))
    lat_new_nc.units =''
    lat_new_nc.description =''
    
    lon_new_nc = ft.createVariable('longitude','f4',('longitude'))
    lon_new_nc.units =''
    lon_new_nc.description =''
    
    vertical_no2_trop_nc = ft.createVariable('vertical_no2_trop','f4',('time','latitude','longitude'))  
    vertical_no2_trop_nc .units =''                                               
    vertical_no2_trop_nc .description =''
    
    qa_nc = ft.createVariable('qa','f4',('time','latitude','longitude'))  
    qa_nc .units =''                                               
    qa_nc .description =''
    
    lat_new_nc[:]               = np.array(lat_new)
    lon_new_nc[:]               = np.array(lon_new)
    vertical_no2_trop_nc[:,:,:] = np.array(trop_no2)
    qa_nc[:,:,:]                = np.array(qa)
    
    ft.close()
























