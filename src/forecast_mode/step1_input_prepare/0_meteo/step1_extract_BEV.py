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

day_list =   ['29','30']

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
                #aod_w = f.variables['aod'][:][0]
                surface_pressure_w = f.variables['pressfc'][:][0]  #unit = Pa
                t2m_w = f.variables['tmp2m'][:][0] #unit = K
                specific_humidity_2m_w = f.variables['spfh2m'][:][0]  #unit = kg/kg-air
                evaporation_w = f.variables['evbs_ave'][:][0]+ f.variables['evcw_ave'][:][0]    #unit = W/m^2
                u10_w = f.variables['ugrd10m'][:][0]  #unit = m/s
                v10_w = f.variables['vgrd10m'][:][0]  #unit = m/s
                pblh_w = f.variables['hpbl'][:][0]    #unit = m
                precipitation_w = f.variables['tprcp'][:][0]  #unit = kg/m^2


                '''
                2)write into a new NetCdf file
                '''
                ft=Dataset('BEV_meteo_'+year+month+day+'_'+filelist[i].replace('aqm.t','').replace('.phy.f0',''),'w',format='NETCDF4')

                time = ft.createDimension('time',None)
                lat  = ft.createDimension('lat',488)
                lon  = ft.createDimension('lon',775)

                lat  = ft.createVariable('lat','f4',('lat','lon'))
                lat.units = f.variables['lat'].units
                lat.description = f.variables['lat'].long_name

                lon  = ft.createVariable('lon','f4',('lat','lon'))
                lon.units = f.variables['lon'].units
                lon.description = f.variables['lon'].long_name

                #aod = ft.createVariable('aod','f4',('time','lat','lon'))
                #aod.units = ''
                #aod.description = 'total aod @ 550nm'

                surface_pressure = ft.createVariable('surface_pressure','f4',('time','lat','lon'))
                surface_pressure.units = 'Pa'
                surface_pressure.description = 'surface_pressure'

                t2m = ft.createVariable('t2m','f4',('time','lat','lon'))
                t2m.units = 'K'
                t2m.description = 't2m'

                specific_humidity_2m = ft.createVariable('specific_humidity_2m','f4',('time','lat','lon'))
                specific_humidity_2m.units = 'kg/kg-air'
                specific_humidity_2m.description = 'specific_humidity_2m'

                evaporation = ft.createVariable('evaporation','f4',('time','lat','lon'))
                evaporation.units = 'W/m^2'
                evaporation.description = 'evaporation bare soil & canopy water'

                u10 = ft.createVariable('u10','f4',('time','lat','lon'))
                u10.units = 'm/s'
                u10.description = 'u10'

                v10 = ft.createVariable('v10','f4',('time','lat','lon'))
                v10.units = 'm/s'
                v10.description = 'v10'

                pblh = ft.createVariable('pblh','f4',('time','lat','lon'))
                pblh.units = 'm'
                pblh.description = 'pblh'

                precipitation = ft.createVariable('precipitation','f4',('time','lat','lon'))
                precipitation.units = 'kg/m^2'
                precipitation.description = 'precipitation'

                '''
                3) assign data
                '''
                lat[:,:]=lat_w
                lon[:,:]=lon_w
                #aod[0,:,:]=aod_w
                surface_pressure[0,:,:]=surface_pressure_w
                t2m[0,:,:] = t2m_w
                specific_humidity_2m[0,:,:]=specific_humidity_2m_w
                evaporation[0,:,:]=evaporation_w
                u10[0,:,:] = u10_w
                v10[0,:,:] = v10_w
                pblh[0,:,:] = pblh_w
                precipitation[0,:,:] = precipitation_w

                ft.close()
                f.close()
