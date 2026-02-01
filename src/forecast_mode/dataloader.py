import pandas as pd
import xarray as xr

import datetime
from datetime import timedelta



import numpy as np
import os
from netCDF4 import Dataset
import fnmatch


class Load_AOD:
    def extract_data(aod_folder,year_list, month_list, day_list):
        "Step 1: Extract data and return as xarray Dataset"
        all_data = []
        times = []
        
        for year in year_list:
            for month in month_list:
                for day in day_list:
                    dir_data = aod_folder+f'/{year}/{year}{month}/{year}{month}{day}/'
                    pattern = 'aqm.t12z.phy.f*'
                    filelist = np.sort(fnmatch.filter(os.listdir(dir_data), pattern))
                    
                    for i in range(24):
                      with Dataset(dir_data + filelist[i], 'r') as f:
                          lat = f.variables['lat'][:]
                          lon = f.variables['lon'][:] - 360
                          aod = f.variables['aod'][:][0]
                          
                          time_str = f"{year}{month}{day}_{filelist[i].replace('aqm.t','').replace('.phy.f0','').replace('.nc','').replace('z','')}"
                          time_str_head= time_str[:-2]
                          time_str_tail= time_str[-2:]
                          
                          time_dt=pd.to_datetime(time_str_head,format='%Y%m%d_%H')+timedelta(hours=int(time_str_tail))
                          times.append(time_dt)
                          all_data.append(aod)
        
        
        'step2.Create xarray Dataset'
        ds = xr.Dataset(
            {
                'aod': (['time', 'lat', 'lon'], np.stack(all_data)),
                'lat': (['lat', 'lon'], lat),
                'lon': (['lat', 'lon'], lon)
            },
            coords={
                'time': times
            }
        )
        return ds


class Load_Meteo:
    def extract_data(meteo_folder,variable_name, year_list, month_list, day_list):
        "Step 1: Extract data and return as xarray Dataset"
        all_data = []
        times = []
        
        for year in year_list:
            for month in month_list:
                for day in day_list:
                    dir_data = meteo_folder+f'/{year}/{year}{month}/{year}{month}{day}/'
                    pattern = 'aqm.t12z.phy.f*'
                    filelist = np.sort(fnmatch.filter(os.listdir(dir_data), pattern))
    
                    for i in range(24):
                        with Dataset(dir_data + filelist[i], 'r') as f:
                            lat = f.variables['lat'][:]
                            lon = f.variables['lon'][:] - 360
                            
                            if variable_name == 'surface_pressure':
                                surface_pressure= f.variables['pressfc'][:][0] #unit = pa
                                all_data.append(surface_pressure)
                                
                            elif variable_name == 'temperature_2m':
                                t2m_w = f.variables['tmp2m'][:][0] #unit = K
                                all_data.append(t2m_w)
                                
                            elif variable_name == 'humidity':
                                specific_humidity_2m_w = f.variables['spfh2m'][:][0]  #unit = kg/kg-air
                                all_data.append(specific_humidity_2m_w)
                                
                            elif variable_name == 'evaporation':
                                evaporation_w = f.variables['evbs_ave'][:][0]+ f.variables['evcw_ave'][:][0]    #unit = W/m^2
                                all_data.append(evaporation_w)
                                
                            elif variable_name == 'wind speed u':
                                u10_w = f.variables['ugrd10m'][:][0]  #unit = m/s
                                all_data.append(u10_w)
                                
                            elif variable_name == 'wind speed v':
                                v10_w = f.variables['vgrd10m'][:][0]  #unit = m/s
                                all_data.append(v10_w)
                                
                            elif variable_name == 'pblh':
                                pblh_w = f.variables['hpbl'][:][0]    #unit = m
                                all_data.append(pblh_w)
                                
                            elif variable_name == 'precipitation':
                                precipitation_w = f.variables['tprcp'][:][0]  #unit = kg/m^2
                                all_data.append(precipitation_w)
                            
                            
                            time_str = f"{year}{month}{day}_{filelist[i].replace('aqm.t','').replace('.phy.f0','').replace('.nc','').replace('z','')}"
                            time_str_head= time_str[:-2]
                            time_str_tail= time_str[-2:]
                            
                            time_dt=pd.to_datetime(time_str_head,format='%Y%m%d_%H')+timedelta(hours=int(time_str_tail))
                            times.append(time_dt)
                            
        
        
        'step2.Create xarray Dataset'
        ds = xr.Dataset(
            {
                variable_name: (['time', 'lat', 'lon'], np.stack(all_data)),
                'lat': (['lat', 'lon'], lat),
                'lon': (['lat', 'lon'], lon)
            },
            coords={
                'time': times
            }
        )
        return ds

class Load_Chem:
    def extract_data(chem_folder,variable_name, year_list, month_list, day_list):
        "Step 1: Extract data and return as xarray Dataset"
        all_data = []
        times = []
        
        for year in year_list:
            for month in month_list:
                for day in day_list:
                    dir_data = chem_folder+f'/{year}/{year}{month}/{year}{month}{day}/'
                    pattern = 'aqm.t12z.chem_sfc.nc'
                    filename = dir_data+pattern
                    f = Dataset(filename,'r')
    
                    for i in range(24):
 
                        lat = f.variables['lat'][0,:,:]
                        lon = f.variables['lon'][0,:,:] - 360
                        
                        if variable_name == 'pm25':                            
                            pm25_w = f.variables['PM25_TOT'][i,0,:,:]
                            all_data.append(pm25_w)
                            
                        elif variable_name == 'o3':                          
                            o3_w = f.variables['o3'][i,0,:,:] 
                            all_data.append(o3_w)
                            

                        time_str = f"{year}{month}{day}"
                        
                        time_dt=pd.to_datetime(time_str,format='%Y%m%d')+timedelta(hours=12+int(i))
                        times.append(time_dt)
                            
        
        
        'step2.Create xarray Dataset'
        ds = xr.Dataset(
            {
                variable_name: (['time', 'lat', 'lon'], np.stack(all_data)),
                'lat': (['lat', 'lon'], lat),
                'lon': (['lat', 'lon'], lon)
            },
            coords={
                'time': times
            }
        )
        return ds


class Load_Static_Data:
    def extract_elevation(static_elevation_data,variable_name,time_length,year_list, month_list, day_list):       
        filename= static_elevation_data
        f = Dataset(filename,'r')
        
        lat= f.variables['lat'][:]
        lon= f.variables['lon'][:]       
        lon2d,lat2d= np.meshgrid(lon,lat)
        
        Data_output=[]
        times= []
        for i in range(time_length):
            data_output_here = f.variables[variable_name][:][0]
            Data_output.append(data_output_here)
            
            time_str = year_list[0]+month_list[0]+day_list[0]
            time_dt=pd.to_datetime(time_str,format='%Y%m%d')+timedelta(hours=12+int(i))
            times.append(time_dt)
        
        ds = xr.Dataset(
            {
                variable_name: (['time', 'lat', 'lon'], np.stack(Data_output)),
                'lat': (['lat', 'lon'], lat2d),
                'lon': (['lat', 'lon'], lon2d)
            },
            coords={
                'time': times
            }
        ) 
            
        return ds
    
    def extract_population(static_population_data,variable_name,time_length,year_list, month_list, day_list):       
        filename= static_population_data
        f = Dataset(filename,'r')
        
        lat= f.variables['lat'][:]
        lon= f.variables['lon'][:]       
        lon2d,lat2d= np.meshgrid(lon,lat)
        
        Data_output=[]
        times=[]
        
        for i in range(time_length):
            data_output_here = f.variables[variable_name][:][0]
            Data_output.append(data_output_here)
            
            time_str = year_list[0]+month_list[0]+day_list[0]
            time_dt=pd.to_datetime(time_str,format='%Y%m%d')+timedelta(hours=12+int(i))
            times.append(time_dt)
        
        ds = xr.Dataset(
            {
                variable_name: (['time', 'lat', 'lon'], np.stack(Data_output)),
                'lat': (['lat', 'lon'], lat2d),
                'lon': (['lat', 'lon'], lon2d)
            },
            coords={
                'time': times
            }
        )   
        return ds
            
    def extract_land_use_cover(static_land_use_cover_data,variable_name,time_length,year_list, month_list, day_list):       
        filename= static_land_use_cover_data
        f = Dataset(filename,'r')
        
        lat= f.variables['lat'][:]
        lon= f.variables['lon'][:]       
        lon2d,lat2d= np.meshgrid(lon,lat)
        
        Data_output=[]
        times=[]
        
        for i in range(time_length):
            data_output_here = f.variables[variable_name][:][0]
            Data_output.append(data_output_here)
            
            time_str = year_list[0]+month_list[0]+day_list[0]
            time_dt=pd.to_datetime(time_str,format='%Y%m%d')+timedelta(hours=12+int(i))
            times.append(time_dt)

        ds = xr.Dataset(
            {
                variable_name: (['time', 'lat', 'lon'], np.stack(Data_output)),
                'lat': (['lat', 'lon'], lat2d),
                'lon': (['lat', 'lon'], lon2d)
            },
            coords={
                'time': times
            }
        )  
            
        return ds
            
    #     return Elevation
    # def extract_anthro_emission():
        # return xxx
    
class Load_Observation:
    def extract_airnow_pm25(obs_folder,obs_filename,start_time,ll_lat,ur_lat,ll_lon,ur_lon): 
        
        end_time = start_time+timedelta(hours=24)
        
        '''
        1) load data from AirNow
        '''
        #obs_filename example: AirNow_20250801_20250831.nc this is downloaded using MELODIES-MONET
        filename = obs_folder+obs_filename
        f1 = xr.open_dataset(filename)
        lat = f1['latitude']       
        lon = f1['longitude']      
        time_ori = f1['time']      
        pm25 = f1['PM2.5'][:,0,:]  
        
        '''
        2) find vacant site
        '''
        site_vacant_list = []
        pm25_sum_by_site = pm25.sum(dim='time')
        for i in range(len(pm25_sum_by_site)):
            if float(pm25_sum_by_site[i]) == -len(time_ori):
                site_vacant_list.append(i)
        
        '''
        3) average by hour
        '''
        pm25_avr_hr = pm25.resample(time = 'h').mean()     
        time_avr_hr = time_ori.resample(time= 'h').mean()    
        
        '''
        4) get useful data
        '''
        pm25_new_list = []
        time_new_list = []
        lat_new_list = []
        lon_new_list = []
        index_new_list = []
        for i in range(len(lat)): 
            if i not in site_vacant_list:
                if float(lat[i]) >= ll_lat and float(lat[i]) <= ur_lat:
                    if float(lon[i]) >= ll_lon and float(lon[i]) <= ur_lon:
                        
                        for j in range(len(time_avr_hr)):  
                            time_here = pd.to_datetime(time_avr_hr[j].values)
                            if time_here >= start_time and time_here <= end_time: 
                                
                                if float(pm25_avr_hr[j][i]) != -1:                                    
                                    pm25_new_list.append(np.round(float(pm25_avr_hr[j][i]),4))                                                         
                                    time_new_list.append(str(time_here))
                                  
                                    lat_new_list.append(float(lat[i]))
                                    lon_new_list.append(float(lon[i]))
                                    index_new_list.append(i)

        return index_new_list,time_new_list,pm25_new_list,lat_new_list,lon_new_list






















































