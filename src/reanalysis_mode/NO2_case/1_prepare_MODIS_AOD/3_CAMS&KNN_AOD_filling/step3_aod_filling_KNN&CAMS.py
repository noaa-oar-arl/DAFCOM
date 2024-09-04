#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:33:31 2022

@author: btang1
"""
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# from pyhdf.SD import SD, SDC
# from mpl_toolkits.basemap import Basemap

from netCDF4 import Dataset
from sklearn.neighbors import KNeighborsRegressor

'''
1) import data
'''
def CountNearbyPoint(AOD_input, i_input,j_input):
    i_start = 0
    i_end = 2873
    j_start = 0
    j_end = 7183
    
    if i_input - 40 > 0:
        i_start = i_input - 40
    if i_input + 40 < 2873:
        i_end = i_input + 40
    if j_input - 40 > 0:
        j_start = j_input - 40
    if j_input + 40 < 7183:
        j_end = j_input + 40    
    
    count = 0
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            if AOD_input[i,j] != 0:
                count += 1
                if count >= 3:
                    break
        if count >= 3:
            break
        
    return count
'''
1-1)get CAMS AOD data
'''
dir_cams = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/data_CONUS_US/1_MODIS_AOD/1_prepare_CAMS_AOD/'
pattern_cams = 'CAMS_AOD_regrid.nc'
filename_cams = dir_cams + pattern_cams
f_cams = Dataset(filename_cams,'r')
AOD_cams = f_cams.variables['cams_aod'][:] #(30,2993,4191)
    
'''
1-2)get MAIAC AOD data
'''
Julian_day_list =['1101','1102','1103','1104','1105','1106','1107','1108','1109','1110',
                  '1111','1112','1113','1114','1115','1116','1117','1118','1119','1120',
                  '1111','1122','1123','1124','1125','1126','1127','1128','1129','1130',] 
Day_index_list =[0+x for x in range(30)]


for kk in range(len(Julian_day_list)):
    dir_ = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/data_CONUS_US/1_MODIS_AOD/2_prepare_MAIAC_AOD/'
    pattern = 'MODIS_AOD_step1_'+str(Julian_day_list[kk])+'.nc'
    filename = dir_ + pattern
    f = Dataset(filename,'r')
    
    
    AOD_MAIAC = f.variables['MODIS_AOD'][0] #(2994,4191)
    latitude = f.variables['latitude'] #2994
    longitude = f.variables['longitude'] #4191
    
    
    
    '''
    2) AOD filling KNN
    '''
    
    '''
    2-1) use pixels has value to build KNN model
    '''
    X = []
    Y = []
    for i in range(len(AOD_MAIAC)): #2994
        for j in range(len(AOD_MAIAC[0])): #4191
            if AOD_MAIAC[i,j] != 0:
                X.append([i,j])
                Y.append(AOD_MAIAC[i][j])
    X = np.array(X)
    Y = np.array(Y)        
    
    # from sklearn.neighbors import KNeighborsRegressor
    knn_model = KNeighborsRegressor(n_neighbors=3,weights ='distance')
    
    knn_model.fit(X, Y)
    
    '''
    2-2) fill AOD for pixels NOT have values
    '''
                
    
    AOD_FILL = np.zeros((2874,7184))
    for i in range(len(AOD_MAIAC)): #2994
        for j in range(len(AOD_MAIAC[0])): #4191
            if AOD_MAIAC[i][j] == 0:
                count_here = CountNearbyPoint(AOD_MAIAC,i,j)
                
                if count_here >= 3:
                    aod_fill_here = knn_model.predict(np.array([[i,j]]))
                    AOD_FILL[i][j] = aod_fill_here[0]
                    print(i,j, 'fill_knn', aod_fill_here[0])
                else:
                    day_index = Day_index_list[kk]                
                    AOD_FILL[i][j] = AOD_cams[day_index][i][j]                                                                    
                    print(i,j, 'fill_cams',AOD_cams[day_index][i][j])
            else:                            
                AOD_FILL[i][j] = AOD_MAIAC[i][j]
                print(i,j, 'no fill', AOD_MAIAC[i][j])
    
    
    AOD_MAIAC_nlayer = AOD_MAIAC.reshape((1,AOD_MAIAC.shape[0],AOD_MAIAC.shape[1])) 
    AOD_fill_nlayer = AOD_FILL.reshape((1,AOD_FILL.shape[0],AOD_FILL.shape[1])) 
    
    
    '''
    3) write in netcdf
    '''
    ft = Dataset((pattern.replace('MODIS_AOD_step1_','AOD_fill_KNN&CAMS_')),'w',format = 'NETCDF4')   
    nlat = ft.createDimension('latitude',2874)
    nlon = ft.createDimension('longitude',7184)
    nlayer = ft.createDimension('time',1)
    
    lat_new = ft.createVariable('latitude','f4',('latitude'))
    lat_new.units =''
    lat_new.description =''
    
    lon_new = ft.createVariable('longitude','f4',('longitude'))
    lon_new.units =''
    lon_new.description =''
    
    variable_new = ft.createVariable('AOD_MAIAC','f4',('time','latitude','longitude'))  
    variable_new .units =''                                               
    variable_new .description =''
    
    variable_new2 = ft.createVariable('AOD_fill_knn&CAMS','f4',('time','latitude','longitude'))  
    variable_new2 .units =''                                               
    variable_new2 .description =''
    
    
    
    lat_new[:] = np.array(latitude)
    lon_new[:] = np.array(longitude)
    variable_new[:,:,:] = np.array(AOD_MAIAC_nlayer)
    variable_new2[:,:,:] = np.array(AOD_fill_nlayer)
    
    ft.close()
        
        
    
        






























