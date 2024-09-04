#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:58:28 2022

@author: btang1
"""


from netCDF4 import Dataset
import numpy as np


date_input_list = [        '0802','0803','0804','0805',
                    '0806','0807','0808','0809','0810',
                    '0811','0812','0813','0814','0815',
                    '0816','0817','0818','0819','0820',
                    '0821','0822','0823','0824','0825',
                    '0826','0827','0828','0829','0830',
                    '0831']

# date_input_list = ['0801']


for date_input in date_input_list:

    
    # date_input = '1104' #change here
    '''
    1) read in land-use .tif data
    '''
    dir_1 = '/Volumes/Ext_Disk_2/Data/satellite_data/MODIS_MAIAC_AOD/2023_08_CONUS_US/'
    
    pattern1 = '2023'+date_input+'_Band1.tif' 
    filename1 = dir_1 + pattern1
    
    pattern2 = '2023'+date_input+'_Band2.tif'
    filename2 = dir_1 + pattern2
    
    pattern3 = '2023'+date_input+'_Band3.tif'
    filename3 = dir_1 + pattern3
    
    from PIL import Image
    im1 = Image.open(filename1)
    img1_array = np.array(im1) 
    # print(img1_array.shape)
    
    im2 = Image.open(filename2)
    img2_array = np.array(im2) 
    # print(img2_array.shape)
    
    
    im3 = Image.open(filename3)
    img3_array = np.array(im3) 
    # print(img3_array.shape)
    
    
    img1_nlayer = img1_array.reshape((1,img1_array.shape[0],img1_array.shape[1])) 
    img2_nlayer = img2_array.reshape((1,img2_array.shape[0],img2_array.shape[1])) 
    img3_nlayer = img3_array.reshape((1,img3_array.shape[0],img3_array.shape[1])) 
    
    img_sum = np.concatenate((img1_nlayer, img2_nlayer, img3_nlayer), axis = 0)
    
    '''
    2) calculate urban only
    '''
    
    '''
    2-1) check is nan
    '''
    # for i in range(len(img1_array)):
    #     for j in range(len(img1_array[0])):
    #         if np.isnan(img1_array[i][j]) == True:
    #             print('1',i,j)
                
    # for i in range(len(img2_array)):
    #     for j in range(len(img2_array[0])):
    #         if np.isnan(img2_array[i][j]) == True:
    #             print('2',i,j)
    
    # for i in range(len(img3_array)):
    #     for j in range(len(img3_array[0])):
    #         if np.isnan(img3_array[i][j]) == True:
    #             print('3',i,j)
                
    Index_matrix =  np.zeros((2874, 7184))         
    AOD_matrix = np.zeros((2874, 7184))     
    
    for i in range(2874):
        for j in range(7184):
            # print(i,j)
            index_here = 0
            aod_here = 0
            for k in range(len(img_sum)):   #THIS IS 3 OR 4
                if img_sum[k][i][j] >= 0:
                    index_here += 1
                    aod_here += img_sum[k][i][j]
                else:
                    index_here += 0
                    aod_here += 0
                    
            if index_here != 0:
                aod_here_in = aod_here/index_here
                AOD_matrix[i][j] = aod_here_in
                Index_matrix[i][j] = index_here
            else:
                AOD_matrix[i][j] = 0
                Index_matrix[i][j] = 0
                    
                    
    
    # print(np.max(AOD_matrix))             
    AOD_matrix = AOD_matrix*0.001
    # print(np.max(AOD_matrix))           
                    
                
    
    AOD_final_nlayer = AOD_matrix.reshape((1,AOD_matrix.shape[0],AOD_matrix.shape[1])) 
    Index_final_nlayer = Index_matrix.reshape((1,Index_matrix.shape[0],Index_matrix.shape[1])) 
    
    
    lat_resolution = (49-25)/2874
    lon_resolution = np.abs((-125+ 65)/7184)   
       
    lat_new = [49- lat_resolution*X for X in range(2874)]
    lon_new = [-125+ lon_resolution *X for X in range(7184)]
    
    '''
    3) write into nc file
    '''
    
    
    ft = Dataset(('MODIS_AOD_step1_'+date_input +'.nc'),'w',format = 'NETCDF4')   #change here
    nlat = ft.createDimension('latitude',2874)
    nlon = ft.createDimension('longitude',7184)
    nlayer = ft.createDimension('time',1)
    
    
    lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
    lat_new_nc.units =''
    lat_new_nc.long_name ='latitude'
    
    lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
    lon_new_nc.units =''
    lon_new_nc.long_name ='longitude'
    
    variable_new = ft.createVariable('MODIS_AOD','f8',('time','latitude','longitude'))  
    variable_new .units =''                                               
    variable_new .description =''
    
    variable2_new = ft.createVariable('index_aod','f8',('time','latitude','longitude'))  
    variable2_new .units =''                                               
    variable2_new .description =''
    
    
    lat_new_nc[:] = np.array(lat_new)
    lon_new_nc[:] = np.array(lon_new)
    variable_new[:,:,:] = np.array(AOD_final_nlayer)
    variable2_new[:,:,:] = np.array(Index_final_nlayer)
    
    
    ft.close()
    
    
    



