import numpy as np
import os
import sys
from netCDF4 import Dataset
import fnmatch

'''
1)read data from WRF-Chen output, BEV_
'''
varibale_name_list=['AOD']

dir_data ='/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/data/1_prepare_aod_gefs/step1_BEV/2023/2023_06/'
pattern ='BEV_*'

wrf_list_raw = fnmatch.filter(os.listdir(dir_data),pattern)
filelist = np.sort(wrf_list_raw)
    
for variable_name in varibale_name_list:    
    ft = Dataset(('Surface_Map_'+variable_name+".nc"),'w',format = 'NETCDF4')
    ntime = ft.createDimension('ntime',len(filelist))
    nlat = ft.createDimension('nlat',721)
    nlon = ft.createDimension('nlon',1440)
    
    time_new = ft.createVariable('time','S1',('ntime',))
    time_new.units =''
    time_new.description =''
    
    lat = ft.createVariable('lat','f4',('nlat','nlon'))
    lat.units =''
    lat.description =''
    
    lon = ft.createVariable('lon','f4',('nlat','nlon'))
    lon.units =''
    lon.description =''
    
    vmrno2 = ft.createVariable(variable_name,'f4',('ntime','nlat','nlon'))
    vmrno2.units =''                                                         
    vmrno2.description =''
        
    for i in range(len(filelist)):
    
        filename = dir_data+filelist[i]
        f = Dataset(filename,'r')
        
        time_w = filelist[i].replace('BEV_','')
        lat_w = f.variables['lat'][:]
        lon_w = f.variables['lon'][:]
        vmrno2_w= f.variables[variable_name][0,:]                                       #change chemical variables here
        
        time_new[:] = time_w
        lat[:,:]= lat_w
        lon[:,:]= lon_w
        vmrno2[i,:,:] = vmrno2_w
        
    
            
        f.close()  
    ft.close()
  
