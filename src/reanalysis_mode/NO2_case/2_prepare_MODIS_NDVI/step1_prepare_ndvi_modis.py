#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:58:28 2022

@author: btang1
"""


from netCDF4 import Dataset
import numpy as np


'''
1) read in land-use .tif data
'''
dir_1 = '/Volumes/Ext_Disk_2/Data/satellite_data/MODIS_NDVI/MODIS_NDVI_2023_08_CONUS_US/'

pattern1 = 'aqua_0815.tif' #change here
filename1 = dir_1 + pattern1

pattern2 = 'terra_0815.tif' #change here
filename2 = dir_1 + pattern2



from PIL import Image
im1 = Image.open(filename1)
img1_array = np.array(im1) #this is the output; xll_corner = 35, yll_corner = 125, 35-40N, 125-130E
print(img1_array.shape)

im2 = Image.open(filename2)
img2_array = np.array(im2) #this is the output; xll_corner = 35, yll_corner = 125, 35-40N, 125-130E
print(img2_array.shape)

'''
2) calculate urban only
'''

'''
2-1) check is nan
'''
# for i in range(len(img2_array)):
#     for j in range(len(img2_array[0])):
#         if np.isnan(img2_array[i][j]) == True:
#             print('1',i,j)
            
# for i in range(len(img2_array)):
#     for j in range(len(img2_array[0])):
#         if np.isnan(img2_array[i][j]) == True:
#             print('2',i,j)
            
NDVI_matrix = np.zeros((2874, 7184))       
for i in range(2874)     :
    for j in range(7184):
        if img1_array[i][j]!= -3000 and img2_array[i][j] != -3000:
            ndvi_avr = (img1_array[i][j] + img2_array[i][j])/2 
            NDVI_matrix[i][j] = ndvi_avr
        else:
            NDVI_matrix[i][j] == 0
       
                
            

NDVI_final_nlayer = NDVI_matrix.reshape((1,NDVI_matrix.shape[0],NDVI_matrix.shape[1])) 



lat_resolution = (49-25)/2874
lon_resolution = np.abs((-125+ 65)/7184)   
   
lat_new = [49- lat_resolution*X for X in range(2874)]
lon_new = [-125+ lon_resolution *X for X in range(7184)]

'''
3) write into nc file
'''


ft = Dataset(('NDVI_step1_0815'+'.nc'),'w',format = 'NETCDF4')   #change here
nlat = ft.createDimension('latitude',2874)
nlon = ft.createDimension('longitude',7184)
nlayer = ft.createDimension('time',1)


lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
lat_new_nc.units =''
lat_new_nc.long_name ='latitude'

lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
lon_new_nc.units =''
lon_new_nc.long_name ='longitude'

variable_new = ft.createVariable('ndvi','f8',('time','latitude','longitude'))  
variable_new .units =''                                               
variable_new .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new[:,:,:] = np.array(NDVI_final_nlayer)

ft.close()






