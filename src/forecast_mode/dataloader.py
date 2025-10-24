#======================
#import library

#======================
import pandas as pd
import xarray as xr

#======================
#main code
#
#======================

# class DataLoader:
#     def __init__(self, config):
#         self.config = config

#     def load(self, file_path):
#         if self.config["data_type"] == "netcdf":
#             return self.read_netcdf(file_path)
#         elif self.config["data_type"] == "csv":
#             return self.read_csv(file_path)
#         else:
#             return ValueError(f"Unsupported data type")

#     def read_netcdf(self, file_path: str) -> xr.Dataset:
#         return xr.open_dataset(file_path)

#     def read_csv(self, file_path: str) -> pd.DataFrame:
#         return pd.read_csv(file_path)

import numpy as np
import os
from netCDF4 import Dataset
import fnmatch

class Load_AOD:
    def extract_data(aod_folder,year_list, month_list, day_list, size_lat_ufs, size_lon_ufs):
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
                            
                            time_str = f"{year}{month}{day}_{filelist[i].replace('aqm.t','').replace('.phy.f0','')}"
                            times.append(time_str)
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
    def extract_data(meteo_folder,variable_name, year_list, month_list, day_list, size_lat_ufs, size_lon_ufs):
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
                            
                            time_str = f"{year}{month}{day}_{filelist[i].replace('aqm.t','').replace('.phy.f0','')}"
                            times.append(time_str)
                            
        
        
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
    def extract_data(chem_folder,variable_name, year_list, month_list, day_list, size_lat_ufs, size_lon_ufs):
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
                            

                        
                        time_str = f"{year}{month}{day}_{filename.replace('aqm.t','').replace('.chem_sfc','')}"
                        times.append(time_str)
                            
        
        
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










