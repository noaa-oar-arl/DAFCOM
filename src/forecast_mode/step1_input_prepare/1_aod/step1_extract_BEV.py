#author:Beiming Tang
#date: Apr 22. 2025


import numpy as np
import os
from netCDF4 import Dataset
import fnmatch

'''
1) read data from UFS-AQM output
'''
year_list  = ['2025']
month_list = ['04']
#day_list =   ['01','02','03','04','05','06','07','08','09','10',
#              '11','12','13','14','15','16','17','18','19','20',
#              '21','22','23','24','25','26','27','28','29','30',
#              '31']

day_list = ['29','30']

for year in year_list:
    for month in month_list:

        for day in day_list:

            dir_data = '/ARLChemWeb/NAQFC/5xpm/'+year+'/'+year+month+'/'+year+month+day+'/'
            pattern= 'aqm.t12z.phy.f*'
            ufs_aqm_list_raw = fnmatch.filter(os.listdir(dir_data),pattern)
            filelist = np.sort(ufs_aqm_list_raw)

            #for i in range(len(filelist)):   #this is 72 hrs forecasts
            for i in range(24): #use only the 24 hours forecast
                filename=dir_data+filelist[i]
                f=Dataset(filename,'r')
    
                lat_w = f.variables['lat'][:]
                lon_w = f.variables['lon'][:] -360
                aod_w = f.variables['aod'][:][0]

                '''
                2)write into a new NetCdf file
                '''
                ft=Dataset('BEV_aod_'+year+month+day+'_'+filelist[i].replace('aqm.t','').replace('.phy.f0',''),'w',format='NETCDF4')

                time = ft.createDimension('time',None)
                lat  = ft.createDimension('lat',488)
                lon  = ft.createDimension('lon',775)

                lat  = ft.createVariable('lat','f4',('lat','lon'))
                lat.units = f.variables['lat'].units
                lat.description = f.variables['lat'].long_name

                lon  = ft.createVariable('lon','f4',('lat','lon'))
                lon.units = f.variables['lon'].units
                lon.description = f.variables['lon'].long_name

                aod = ft.createVariable('aod','f4',('time','lat','lon'))
                aod.units = ''
                aod.description = 'total aod @ 550nm'

                '''
                3) assign data
                '''
                lat[:,:]=lat_w
                lon[:,:]=lon_w
                aod[0,:,:]=aod_w

                ft.close()
                f.close()
