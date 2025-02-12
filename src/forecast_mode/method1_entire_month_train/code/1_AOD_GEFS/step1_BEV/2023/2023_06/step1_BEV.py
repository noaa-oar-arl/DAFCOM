import numpy as np
from netCDF4 import Dataset
import xarray as xr

'''
1) open GEFS AOD data,using grib2io (updated by Dr.Barry Baker from NOAA ARL)
'''
date_list=['0601','0602','0603','0604','0605','0606','0607','0608','0609','0610',
           '0611','0612','0613','0614','0615','0616','0617','0618','0619','0620',
           '0621','0622','0623','0624','0625','0626','0627','0628','0629','0630']
for date in date_list:
    dir_data ='/groups/ESS3/btang6/DATA/reanalysis_data/GEFS_AOD/chem_'+date+'/pgrb2ap25/'
    pattern_list =['gefs.chem.t00z.a2d_0p25.f000.grib2', 'gefs.chem.t00z.a2d_0p25.f003.grib2', 
                   'gefs.chem.t00z.a2d_0p25.f006.grib2', 'gefs.chem.t00z.a2d_0p25.f009.grib2', 
                   'gefs.chem.t00z.a2d_0p25.f012.grib2', 'gefs.chem.t00z.a2d_0p25.f015.grib2',
                   'gefs.chem.t00z.a2d_0p25.f018.grib2', 'gefs.chem.t00z.a2d_0p25.f021.grib2']

    for pattern in pattern_list:
        filename= dir_data+pattern

        y= xr.open_dataset(filename,engine='grib2io',filters=dict(typeOfFirstFixedSurface=10))
        #print(y)

        lat_w = y['latitude'].values
        lon_w = y['longitude'].values
        AOD_w = y['totAOD550'].values


        '''
        2) write into a netcdf file
        '''
        ft = Dataset('BEV_'+date+'_'+pattern.replace('gefs.chem.t00z.a2d_0p25.f','').replace('.grib2','')+'.nc','w',format = 'NETCDF4')

        ntime = ft.createDimension('ntime',None)
        nlat = ft.createDimension('nlat',721)
        nlon = ft.createDimension('nlon',1440)

        lat = ft.createVariable('lat','f4',('nlat','nlon'))
        lon = ft.createVariable('lon','f4',('nlat','nlon'))
        AOD = ft.createVariable('AOD','f4',('ntime','nlat','nlon'))

        lat[:,:]= lat_w
        lon[:,:]= lon_w
        AOD[0,:,:] = AOD_w

        ft.close()










