#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:58:28 2022

@author: btang1
"""


from netCDF4 import Dataset
import numpy as np

'''
0) initialization
'''

# input_pattern = 'srtm_18_07/srtm_18_07.tif'
# start_lat = 30
# start_lon = -95
# output_name = '1807'

def GetInput(column_index_input, row_index_input):
    column_index_list = [12,13,14,15,16,17,18,19,20,21,22,23]
    lon_start_value_list = [-125,-120,-115,-110,-105,-100,-95,-90,-85,-80,-75,-70]
    row_index_list = [3,4,5,6,7]
    lat_start_value_list = [50,45,40,35,30]
    
    for i in range(len(column_index_list)):
        if column_index_list[i] == column_index_input:
            output_lon_value = lon_start_value_list[i]

    
    for j in range(len(row_index_list)):
        if row_index_list[j] == row_index_input:
            output_lat_value = lat_start_value_list[j]
    
    output_input_pattern = 'srtm_'+str(column_index_input)+'_0'+str(row_index_input)+'/srtm_'+str(column_index_input)+'_0'+str(row_index_input)+'.tif'
    output_start_lat = output_lat_value
    output_start_lon = output_lon_value
    output_name = str(column_index_input)+'0'+str(row_index_input)
    
    return output_input_pattern, output_start_lat,output_start_lon, output_name

# list1=[12]
# list2=[3,4,5,6]

list1=[13,14,15,16,17,18,19,20,21]
list2=[3,4,5,6,7]

# list1=[22]
# list2=[3,4,5]

# list1=[23]
# list2=[3,4]

for i in list1:
    for j in list2:
        
        print(i,j)
        input_pattern, start_lat, start_lon, output_name = GetInput(i,j)
        
        
        
        
        
        '''
        1) read in land-use .tif data
        '''
        dir_1 = '/Volumes/Ext_Disk_2/Data/reanalysis_data/SRTM_surface_elevation_data/CONUS_US/'
        
        pattern1 = input_pattern
        filename1 = dir_1 + pattern1
        
        
        from PIL import Image
        im1 = Image.open(filename1)
        img1_array = np.array(im1) #this is the output; xll_corner = 35, yll_corner = 125, 35-40N, 125-130E
        print(img1_array.shape)
        
        '''
        2)kick out nan values
        '''
        def KickImage(input_image):
            output_image = np.zeros((6000,6000))
            for i in range(6000):
                for j in range(6000):
                    
                    if input_image[i][j] == -32768:
                        # print(i,j,'ocean')
                        output_image[i][j] = 0
                    else:
                        # print(i,j)
                        output_image[i][j] = input_image[i][j]
            return output_image
        
        output_image1 = KickImage(img1_array)
        output_image1_nlayer = output_image1 .reshape((1,output_image1 .shape[0],output_image1 .shape[1]))
        
        
        lat_new = [start_lat- 0.00083333*X for X in range(6000)]
        lon_new = [start_lon+ 0.00083333*X for X in range(6000)]
        
        '''
        3) write into nc file
        '''
        
        
        ft = Dataset(('Elevation_step1_'+output_name +'.nc'),'w',format = 'NETCDF4')   
        nlat = ft.createDimension('latitude',6000)
        nlon = ft.createDimension('longitude',6000)
        nlayer = ft.createDimension('time',1)
        
        
        lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
        lat_new_nc.units =''
        lat_new_nc.long_name ='latitude'
        
        lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
        lon_new_nc.units =''
        lon_new_nc.long_name ='longitude'
        
        variable_new = ft.createVariable('Elevation','f8',('time','latitude','longitude'))  
        variable_new .units =''                                               
        variable_new .description =''
        
        lat_new_nc[:] = np.array(lat_new)
        lon_new_nc[:] = np.array(lon_new)
        variable_new[:,:,:] = np.array(output_image1_nlayer)
        
        ft.close()
        
        
        
        
        






