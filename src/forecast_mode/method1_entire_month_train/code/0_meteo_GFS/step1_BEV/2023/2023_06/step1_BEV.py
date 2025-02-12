import numpy as np
import os
#import sys
from netCDF4 import Dataset
import fnmatch

'''
1)read data from WRF-Chen output
'''
dir_data ='/groups/ESS3/btang6/DATA/reanalysis_data/GFS_Meterology/CONUS/2023/2023_06/'
pattern ='2023_*'

wrf_list_raw = fnmatch.filter(os.listdir(dir_data),pattern)
filelist = np.sort(wrf_list_raw)

for i in range(len(filelist)):
    filename = dir_data+filelist[i]
    f = Dataset(filename,'r')

    lat_w = f.variables['lat'][:]
    lon_w = f.variables['lon'][:]-360
    surface_pressure_w = f.variables['pressfc'][:][0]  #unit = Pa
    t2m_w = f.variables['tmp2m'][:][0]                 #unit = K
    specific_humidity_2m_w = f.variables['spfh2m'][:][0]    #unit = kg/kg
    evaporation_w = f.variables['evbs_ave'][:][0]        #unit = W/m^2
    u10_w = f.variables['ugrd10m'][:][0]             #unit = m/s
    v10_w = f.variables['vgrd10m'][:][0]             #unit = m/s
    pblh_w = f.variables['hpbl'][:][0]               #unit =m
    precipitation_w = f.variables['tprcp'][:][0]     #unit = kg/m^2
    
    '''
    2)write a new netCDF
    '''
    ft = Dataset('BEV_'+filelist[i],'w',format = 'NETCDF4')
     
    ntime = ft.createDimension('ntime',None)
    nlat = ft.createDimension('nlat',1536)
    nlon = ft.createDimension('nlon',3072)
    
    lat = ft.createVariable('lat','f4',('nlat','nlon'))
    lat.units =f.variables['lat'].units
    lat.description =f.variables['lat'].long_name
    
    lon = ft.createVariable('lon','f4',('nlat','nlon'))
    lon.units =f.variables['lon'].units
    lon.description =f.variables['lon'].long_name

    surface_pressure = ft.createVariable('surface_pressure','f4',('ntime','nlat','nlon'))
    surface_pressure.units ='Pa'
    surface_pressure.description ='surface_pressure'
    
    t2m = ft.createVariable('t2m','f4',('ntime','nlat','nlon'))
    t2m.units = 'K'
    t2m.description = 't2m'
    
    specific_humidity_2m = ft.createVariable('specific_humidity_2m','f4',('ntime','nlat','nlon'))
    specific_humidity_2m.units = 'kg/kg'
    specific_humidity_2m.description = 'specific_humidity_2m'

    evaporation = ft.createVariable('evaporation','f4',('ntime','nlat','nlon'))
    evaporation.units = 'W/m^2'
    evaporation.description = 'evaporation'

    u10 = ft.createVariable('u10','f4',('ntime','nlat','nlon'))
    u10.units = 'm/s'
    u10.description = 'u10'

    v10 = ft.createVariable('v10','f4',('ntime','nlat','nlon'))
    v10.units = 'm/s'
    v10.description = 'v10'

    pblh = ft.createVariable('pblh','f4',('ntime','nlat','nlon'))
    pblh.units = 'm'
    pblh.description = 'pblh'

    precipitation = ft.createVariable('precipitation','f4',('ntime','nlat','nlon'))
    precipitation.units = 'kg/m^2'
    precipitation.description = 'precipitation'

    '''
    3)input data
    '''
    lat[:,:]= lat_w
    lon[:,:]= lon_w
    surface_pressure[0,:,:] = surface_pressure_w
    t2m[0,:,:] = t2m_w
    specific_humidity_2m[0,:,:] = specific_humidity_2m_w
    evaporation[0,:,:] = evaporation_w
    u10[0,:,:] = u10_w
    v10[0,:,:] = v10_w
    pblh[0,:,:] = pblh_w
    precipitation[0,:,:] = precipitation_w
       
    ft.close()
    f.close()



