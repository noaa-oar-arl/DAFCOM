#author:Beiming Tang
#date: Apr 24. 2025


import numpy as np
#import os
from netCDF4 import Dataset
#import fnmatch

'''
1) read data from UFS-AQM output
'''
year_list  = ['2025']
month_list = ['04']

day_list =   ['29','30']

for year in year_list:
    for month in month_list:

        for day in day_list:

            dir_data = '/ARLChemWeb/NAQFC/5xpm/'+year+'/'+year+month+'/'+year+month+day+'/'
            pattern= 'aqm.t12z.chem_sfc.nc'
            filename = dir_data+pattern
            f = Dataset(filename,'r')

            for i in range(24): #total 72, use the 1st 24 hrs.
                lat_w = f.variables['lat'][0,:,:]
                lon_w = f.variables['lon'][0,:,:] -360
                pm25_w = f.variables['PM25_TOT'][i,0,:,:]
                o3_w = f.variables['o3'][i,0,:,:] 
                no2_w = f.variables['no2'][i,0,:,:] 
                no_w = f.variables['no'][i,0,:,:]  
                nox_w = f.variables['no2'][i,0,:,:]+ f.variables['no'][i,0,:,:]


                '''
                2)write into a new NetCdf file
                '''
                if i <10:
                    ft=Dataset('BEV_chem_'+year+month+day+'_0'+str(i)+'.nc','w',format='NETCDF4')
                else:
                    ft=Dataset('BEV_chem_'+year+month+day+'_'+str(i)+'.nc','w',format='NETCDF4')

                time = ft.createDimension('time',None)
                lat  = ft.createDimension('lat',488)
                lon  = ft.createDimension('lon',775)

                lat  = ft.createVariable('lat','f4',('lat','lon'))
                lat.units = f.variables['lat'].units
                lat.description = f.variables['lat'].long_name

                lon  = ft.createVariable('lon','f4',('lat','lon'))
                lon.units = f.variables['lon'].units
                lon.description = f.variables['lon'].long_name

                pm25 = ft.createVariable('pm25','f4',('time','lat','lon'))
                pm25.units = 'ug/m^3'
                pm25.description = ''

                o3  = ft.createVariable('o3','f4',('time','lat','lon'))
                o3.units = 'ppbv'
                o3.description = ''

                no2 = ft.createVariable('no2','f4',('time','lat','lon'))
                no2.units = 'ppbv'
                no2.description = ''

                no = ft.createVariable('no','f4',('time','lat','lon'))
                no.units = 'ppbv'
                no.description = ''

                nox = ft.createVariable('nox','f4',('time','lat','lon'))
                nox.units = 'ppbv'
                nox.description = ''

                '''
                3) assign data
                '''
                lat[:,:]= lat_w
                lon[:,:]= lon_w
                pm25[0,:,:]= pm25_w
                o3[0,:,:]= o3_w
                no2[0,:,:]= no2_w
                no[0,:,:]= no_w
                nox[0,:,:]= nox_w

                ft.close()
            f.close()
