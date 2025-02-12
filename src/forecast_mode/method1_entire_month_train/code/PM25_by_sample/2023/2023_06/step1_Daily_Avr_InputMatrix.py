#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:07:08 2022

@author: btang1
"""

import numpy as np
from netCDF4 import Dataset

'''
0) initialization
'''

dir_ = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/data/'
dir_year = '2023'
dir_month = '2023_06'
number_of_days = 30
dir_year_nemo = '2019'
dir_month_nemo = '2019_06'
start_julian_day = 152 #182 for july, 213 for august

print('this is 2023_06 input data')
'''
0) Meteorology
'''
print('0 meteo')
pattern_0 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_pblh.nc'
filename_0 = dir_ + pattern_0
f0 = Dataset(filename_0,'r')
LAT = f0.variables['lat'][:] #2400
LON = f0.variables['lon'][:] #6000

pattern_1 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_pblh.nc'
filename_1 = dir_ + pattern_1
f1 = Dataset(filename_1,'r')
blh = f1.variables['pblh'][:] 

pattern_2 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_specific_humidity_2m.nc'
filename_2 = dir_ + pattern_2
f2 = Dataset(filename_2,'r')
d2m = f2.variables['specific_humidity_2m'][:] 

pattern_3 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_evaporation.nc'
filename_3 = dir_ + pattern_3
f3 = Dataset(filename_3,'r')
e = f3.variables['evaporation'][:]

pattern_4 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_surface_pressure.nc'
filename_4 = dir_ + pattern_4
f4 = Dataset(filename_4,'r')
sp = f4.variables['surface_pressure'][:]

pattern_5 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_t2m.nc'
filename_5 = dir_ + pattern_5
f5 = Dataset(filename_5,'r')
t2m = f5.variables['t2m'][:]

pattern_6 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_precipitation.nc'
filename_6 = dir_ + pattern_6
f6 = Dataset(filename_6,'r')
tp = f6.variables['precipitation'][:]

pattern_7 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_u10.nc'
filename_7 = dir_ + pattern_7
f7 = Dataset(filename_7,'r')
u10 = f7.variables['u10'][:]

pattern_8 = '0_prepare_meteo_GFS/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GFS_meteo_0p01_v10.nc'
filename_8 = dir_ + pattern_8
f8 = Dataset(filename_8,'r')
v10 = f8.variables['v10'][:]



BLH =blh
D2M = d2m
E = e
SP = sp
T2M = t2m
TP = tp
U10 = u10
V10 = v10



'''
1)AOD
'''
print('1 aod')
AOD = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):    
    pattern_9 = '1_prepare_aod_gefs/step4_regrid_0p01/'+dir_year+'/'+dir_month+'/GEFS_0p01_AOD.nc'
    filename_9 = dir_ + pattern_9
    f9 = Dataset(filename_9,'r')
    aod = f9.variables['AOD'][:][i]
    AOD.append(aod)
#print('AOD', np.shape(AOD))

'''
2) land use cover
'''
print('2 land use conver')
Land_Use_Cover = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    
    pattern_11 = '2_prepare_luc_modis/step2_regrid_0p01/'+dir_year+'/LUC_0p01_regrid_data.nc'
    filename_11 = dir_ + pattern_11
    f11 = Dataset(filename_11,'r')
    land_use_cover = f11.variables['luc'][:][0] 
    Land_Use_Cover.append(land_use_cover)
#print('land cover use:',np.shape(Land_Use_Cover))
    
    

'''
3) elevations
'''
print('3 elevation')
Elevation = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    
    pattern_12 = '3_prepare_elevation/step4_regrid_0p01/elevation_final_regrid_0p01.nc'
    filename_12 = dir_ + pattern_12
    f12 = Dataset(filename_12,'r')
    elevation = f12.variables['Elevation'][:][0] 
    Elevation.append(elevation)
# print('elevation:',np.shape(Elevation))    
    


'''
4) populations
'''
print('4 population')
Population = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    
    pattern_13 = '4_prepare_population/step2_regrid_0p01/'+dir_year+'/Population_0p01_regrid_data.nc'
    filename_13 = dir_ + pattern_13
    f13 = Dataset(filename_13,'r')
    population = f13.variables['Population'][:][0] 
    Population.append(population)
# print('population:',np.shape(Population))      
    
    
 
'''
5) This section is for UFS AQM Result as input
'''  
print('5 UFS AQM')  
pattern_14 = '5_prepare_UFS_AQM/step3_regrid_0p01/'+dir_year+'/'+dir_month+'/'+'UFS_0p01_regrid_data.nc'
filename_14 = dir_+ pattern_14
f14 = Dataset(filename_14,'r')

UFS_AQM_PM25_daily = f14.variables['pm25'][:] 
# print('CAMS PM25',np.shape(CAMS_pm25_daily))    



'''
6) NEMO 2019 emissions
'''
print('6 emissions anthropogenic NEMO')

pattern_18 = '6_prepare_emis_NEMO/step2_0p01_final/'+dir_year_nemo+'/'+dir_month_nemo+'/E_bc_0p01_regrid_data.nc'
filename_18 = dir_ + pattern_18
f18 = Dataset(filename_18,'r')
E_BC_daily = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    E_BC = f18.variables['E_bc'][:][0]
    E_BC_daily.append(E_BC)


pattern_19 = '6_prepare_emis_NEMO/step2_0p01_final/'+dir_year_nemo+'/'+dir_month_nemo+'/E_nh3_0p01_regrid_data.nc'
filename_19 = dir_ + pattern_19
f19 = Dataset(filename_19,'r')
E_NH3_daily = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    E_NH3 = f19.variables['E_nh3'][:][0]
    E_NH3_daily.append(E_NH3)

pattern_20 = '6_prepare_emis_NEMO/step2_0p01_final/'+dir_year_nemo+'/'+dir_month_nemo+'/E_nox_0p01_regrid_data.nc'
filename_20 = dir_ + pattern_20
f20 = Dataset(filename_20,'r')
E_NOX_daily = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    E_NOX = f20.variables['E_nox'][:][0]
    E_NOX_daily.append(E_NOX)

pattern_21 = '6_prepare_emis_NEMO/step2_0p01_final/'+dir_year_nemo+'/'+dir_month_nemo+'/E_voc_0p01_regrid_data.nc'
filename_21 = dir_ + pattern_21
f21 = Dataset(filename_21,'r')
E_VOC_daily = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    E_VOC = f21.variables['E_voc'][:][0]
    E_VOC_daily.append(E_VOC)

pattern_22 = '6_prepare_emis_NEMO/step2_0p01_final/'+dir_year_nemo+'/'+dir_month_nemo+'/E_pm25_0p01_regrid_data.nc'
filename_22 = dir_ + pattern_22
f22 = Dataset(filename_22,'r')
E_PM25_daily = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    E_PM25 = f22.variables['E_pm25'][:][0]
    E_PM25_daily.append(E_PM25)

pattern_23 = '6_prepare_emis_NEMO/step2_0p01_final/'+dir_year_nemo+'/'+dir_month_nemo+'/E_so2_0p01_regrid_data.nc'
filename_23 = dir_ + pattern_23
f23 = Dataset(filename_23,'r')
E_SO2_daily = []
days_list = [1+X for X in range(number_of_days)]
for i in range(len(days_list)):
    E_SO2= f23.variables['E_so2'][:][0]
    E_SO2_daily.append(E_SO2)

'''
7) get LAT and Lon and distance
'''

'''
7-1) Lon
'''
print('7 prepare lat and lon')
Lon_matrix = []
for i in range(2400):
    Lon_matrix.append(LON)    
Lon_matrix = np.reshape(Lon_matrix,(2400,6000))

Lon_matrix_daily = []
for i in range(number_of_days):
    Lon_matrix_daily.append(Lon_matrix)

'''
7-2) Lat
'''
Lat_matrix = []
for i in range(6000):
    Lat_matrix.append(np.transpose(LAT))      
Lat_matrix = np.reshape(Lat_matrix,(6000,2400))
Lat_matrix = np.transpose(Lat_matrix)

Lat_matrix_daily = []
for i in range(number_of_days):
    Lat_matrix_daily.append(Lat_matrix)


'''
7-3) Distance 1,2,3,4,5
'''
print('7 prepare distance')
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

D1 = [] #north west
D2 = [] #north east
D3 = [] #south west
D4 = [] #south east
D5 = [] #center
for i in range(2400):
    for j in range(6000):
        # print(i,j)
        d1 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-125, 49)
        d2 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-65, 49)
        d3 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-125, 25)
        d4 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-65, 25)
        d5 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-90, 37)      
        D1.append(d1)
        D2.append(d2)
        D3.append(d3)
        D4.append(d4)
        D5.append(d5)

D1 = np.reshape(D1, (2400,6000))
D2 = np.reshape(D2, (2400,6000))
D3 = np.reshape(D3, (2400,6000))
D4 = np.reshape(D4, (2400,6000))
D5 = np.reshape(D5, (2400,6000))

D1_matrix_daily = []
for i in range(number_of_days):
    D1_matrix_daily.append(D1)
D2_matrix_daily = []
for i in range(number_of_days):
    D2_matrix_daily.append(D2)
D3_matrix_daily = []
for i in range(number_of_days):
    D3_matrix_daily.append(D3)
D4_matrix_daily = []
for i in range(number_of_days):
    D4_matrix_daily.append(D4)
D5_matrix_daily = []
for i in range(number_of_days):
    D5_matrix_daily.append(D5)
    
'''
8) time
'''
print('8 prepare julian day')
days_list = [start_julian_day+X for X in range(number_of_days)]

def GetDayMatrix(day_input):
    day_matrix = []
    for i in range(14400000):
        day_matrix.append(day_input)
    day_matrix = np.reshape(day_matrix,(2400,6000))
    return day_matrix

day_matrix_daily = []
for i in range(number_of_days):
    day_here = days_list[i]
    day_matrix_here = GetDayMatrix(day_here)
    day_matrix_daily.append(day_matrix_here)
    
    
    
     
    
    
    
    
    
    






    
    
    
    
    
    
    
    
    
    
    
