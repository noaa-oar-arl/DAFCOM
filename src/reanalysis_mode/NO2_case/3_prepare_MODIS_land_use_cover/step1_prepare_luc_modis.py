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
dir_1 = '/Volumes/Ext_Disk_2/Data/satellite_data/MODIS_LAND_USE_COVER/2023_06_CONUS_US/'
pattern1 = 'luc_2023_type5.tif'#25-49 N, -125- -65W
filename1 = dir_1 + pattern1





from PIL import Image
im1 = Image.open(filename1)
img1_array = np.array(im1) #this is the output; xll_corner = 35, yll_corner = 125, 35-40N, 125-130E
print(img1_array.shape)



'''
2) calculate urban only
'''
LUC_final = np.zeros((5748, 14368))
for i in range(len(img1_array)):
    for j in range(len(img1_array[0])):
        
        if img1_array[i][j] == 9:
            LUC_final[i][j] = 1
            # print(i,j)
        else:
            LUC_final[i][j] = 0
                
            

LUC_final_nlayer = LUC_final.reshape((1,LUC_final.shape[0],LUC_final.shape[1])) 



lat_resolution = (49-25)/5748
lon_resolution = np.abs((-125+ 65)/14368)   
   
lat_new = [49- lat_resolution*X for X in range(5748)]
lon_new = [-125+ lon_resolution *X for X in range(14368)]

'''
3) write into nc file
'''


ft = Dataset(('LUC_step1_data'+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',5748)
nlon = ft.createDimension('longitude',14368)
nlayer = ft.createDimension('time',1)


lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
lat_new_nc.units =''
lat_new_nc.long_name ='latitude'

lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
lon_new_nc.units =''
lon_new_nc.long_name ='longitude'

variable_new = ft.createVariable('luc','f8',('time','latitude','longitude'))  
variable_new .units =''                                               
variable_new .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new[:,:,:] = np.array(LUC_final_nlayer)

ft.close()






