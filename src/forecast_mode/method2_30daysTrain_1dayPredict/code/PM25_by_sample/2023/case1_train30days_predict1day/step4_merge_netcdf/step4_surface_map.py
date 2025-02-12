import numpy as np
import os
import sys
from netCDF4 import Dataset
import fnmatch

'''
1)read data from WRF-Chen output, BEV_
'''
variable_name='pm25'
month = '2023_08'

dir_data ='/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method2_30daysTrain_1dayPredict/data/2023/step3_generate_netcdf/'+month+'/'
pattern ='forecast_*'

wrf_list_raw = fnmatch.filter(os.listdir(dir_data),pattern)
filelist = np.sort(wrf_list_raw)
    
ft = Dataset(('Surface_Map_'+variable_name+'_0p01_'+month+".nc"),'w',format = 'NETCDF4')
ntime = ft.createDimension('ntime',len(filelist))
nlat = ft.createDimension('nlat',2400)
nlon = ft.createDimension('nlon',6000)
    
lat = ft.createVariable('lat','f4',('nlat'))
lat.units =''
lat.description =''
    
lon = ft.createVariable('lon','f4',('nlon'))
lon.units =''
lon.description =''
    
variable = ft.createVariable(variable_name,'f4',('ntime','nlat','nlon'))
variable.units =''                                                         
variable.description =''
        
for i in range(len(filelist)):
    
    filename = dir_data+filelist[i]
    f = Dataset(filename,'r')
        
    lat_w = f.variables['latitude'][:]
    lon_w = f.variables['longitude'][:]
    variable_w= f.variables[variable_name][0,:]
    print(np.shape(variable_w))

    lat[:]= lat_w
    lon[:]= lon_w
    variable[i,:,:] = variable_w
        
    
            
    f.close()  
ft.close()
  
