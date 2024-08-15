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
dir_1 = '/Volumes/Ext_Disk_2/Data/reanalysis_data/LandScan_population_data/2022/'
pattern = 'CONUS_population_2022.tif'

filename = dir_1 + pattern

from PIL import Image
im = Image.open(filename)
# im.show()



'''
2) check dimension, and minimum value
'''
img1_array = np.array(im) #this is the output; xll_corner = 25, yll_corner = -90, 25-50N, 125-90W
print(img1_array.shape) #3000,7200
print(np.min(img1_array))
print(np.max(img1_array))


'''
3) for negative population value, set them to 0
'''
output_population = np.zeros((2880,7200))
for i in range(2880):
    for j in range(7200):
        if img1_array[i][j] <0:
            print(i,j)
            output_population[i][j] = 0
        else:
            output_population[i][j] = img1_array[i][j]
            

output_population_nlayer = output_population.reshape((1,output_population.shape[0],output_population.shape[1]))
    
    
lat_new = [49- 0.0083333*X for X in range(2880)]
lon_new = [-125+ 0.0083333*X for X in range(7200)]
# lon_new_, lat_new_ = np.meshgrid(lon_new, lat_new)
                

'''
2) write into nc file
'''


ft = Dataset(('Population_step1_data'+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2880)
nlon = ft.createDimension('longitude',7200)
nlayer = ft.createDimension('time',1)


lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
lat_new_nc.units =''
lat_new_nc.long_name ='latitude'

lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
lon_new_nc.units =''
lon_new_nc.long_name ='longitude'

variable_new = ft.createVariable('Population','f8',('time','latitude','longitude'))  
variable_new .units =''                                               
variable_new .description =''

lat_new_nc[:] = np.array(lat_new)
lon_new_nc[:] = np.array(lon_new)
variable_new[:,:,:] = np.array(output_population_nlayer)

ft.close()






