#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 19:33:31 2022

@author: btang1
"""
import numpy as np
from netCDF4 import Dataset
import math

'''
1)get NO2 TROPOMI data
'''
Date_list = ['0801','0802','0803','0804','0805','0806','0807','0808','0809','0810',
              '0811','0812','0813','0814','0815','0816','0817','0818','0819','0820',
              '0821','0822','0823','0824','0825','0826','0827','0828','0829','0830',
              '0831']

# Date_list = ['0811']

for iii in range(len(Date_list)):
    dir_no2 = '/Volumes/Ext_Disk_2/Data/satellite_data/TROPOMI_NO2/2023_08/'
    pattern_no2 = 'tropomi_v0204_OFFL_NO2_TropVCD_005degree_CONUS_QA075_20230801T20230901.nc'
    filename_no2 = dir_no2 + pattern_no2
    f_no2 = Dataset(filename_no2,'r')
    NO2_trop = f_no2.variables['TROPOMI_NO2_TropVCD'][iii+10] 
      
    LAT_no2 = f_no2.variables['lat'][:]
    LON_no2 = f_no2.variables['lon'][:]
    
    # for i in range(len(LAT_no2)):
    #     for j in range(len(LON_no2)):
    #         if math.isnan(NO2_trop[i][j]) == True:
    #             print(i,j)
        
    '''
    2) filling KNN
    '''
    
    '''
    2-1) use pixels has value to build KNN model
    '''
    X = []
    Y = []
    for i in range(len(NO2_trop)): #540
        for j in range(len(NO2_trop[0])): #1200
            if math.isnan(NO2_trop[i][j]) == False:
                # print('have no nan value')
                X.append([i,j])
                Y.append(NO2_trop[i][j])
    X = np.array(X)
    Y = np.array(Y)        
    
    from sklearn.neighbors import KNeighborsRegressor
    knn_model = KNeighborsRegressor(n_neighbors=3,weights ='distance')
    
    knn_model.fit(X, Y)
    
    '''
    2-2) fill AOD for pixels NOT have values
    '''
                
    
    NO2_FILL = np.zeros((540,1200))
    for i in range(len(NO2_trop)): #540
        for j in range(len(NO2_trop[0])): #1200
            if math.isnan(NO2_trop[i][j]) == True:
    
                no2_fill_here = knn_model.predict(np.array([[i,j]]))
                NO2_FILL[i][j] = no2_fill_here[0]
                print(i,j, 'fill_knn', no2_fill_here[0])
    
            else:                            
                NO2_FILL[i][j] = NO2_trop[i][j]
                # print(i,j, 'no fill', NO2_cams[i][j])
    
    
    NO2_trop_nlayer = NO2_trop.reshape((1,NO2_trop.shape[0],NO2_trop.shape[1])) 
    NO2_fill_nlayer = NO2_FILL.reshape((1,NO2_FILL.shape[0],NO2_FILL.shape[1])) 
    
    print('finish data prepare')
    '''
    3) write in netcdf
    '''
    ft = Dataset(('NO2_TROP_'+Date_list[iii]+'.nc'),'w',format = 'NETCDF4')   
    nlat = ft.createDimension('latitude',540)
    nlon = ft.createDimension('longitude',1200)
    nlayer = ft.createDimension('time',1)
    
    lat_new = ft.createVariable('latitude','f4',('latitude'))
    lat_new.units =''
    lat_new.description =''
    
    lon_new = ft.createVariable('longitude','f4',('longitude'))
    lon_new.units =''
    lon_new.description =''
    
    variable_new = ft.createVariable('NO2_TROP_ORI','f4',('time','latitude','longitude'))  
    variable_new .units =''                                               
    variable_new .description =''
    
    variable_new2 = ft.createVariable('NO2_TROP_ORI_KNNFill','f4',('time','latitude','longitude'))  
    variable_new2 .units =''                                               
    variable_new2 .description =''
    
    
    
    lat_new[:] = np.array(LAT_no2)
    lon_new[:] = np.array(LON_no2)
    variable_new[:,:,:] = np.array(NO2_trop_nlayer)
    variable_new2[:,:,:] = np.array(NO2_fill_nlayer)
    
    ft.close()
        
        
    
        






























