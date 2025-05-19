'''
author: Beiming.Tang
date: 05/15/2025
'''
import numpy as np
from netCDF4 import Dataset
from datetime import datetime as dt
from datetime import timedelta
import xarray as xr

'''
0) initialization
'''

dir_= '/data/aqf3/beiming.tang/DAFCOM/data/'
dir_year='2025'
dir_month='2025_04'#change1
date = 27          #change2
start_hour = 12    #change3


number_of_hours= 96
hour_start = (date-1)*24
hour_end = (date+3)*24

'''
1) meteorology
'''
print('0 meteo')

pattern_0 = '0_meteo/'+dir_year+'/'+dir_month+'/evaporation_0p01.nc'
filename_0 = dir_+pattern_0
f0= xr.open_dataset(filename_0)
LAT = f0.variables['lat'][:]
LON = f0.variables['lon'][:]

pattern_1= '0_meteo/'+dir_year+'/'+dir_month+'/pblh_0p01.nc'
filename_1 = dir_+pattern_1
f1= xr.open_dataset(filename_1)
blh=f1.variables['pblh'][hour_start:hour_end,:,:]

pattern_2= '0_meteo/'+dir_year+'/'+dir_month+'/specific_humidity_2m_0p01.nc'
filename_2 = dir_+pattern_2
f2= xr.open_dataset(filename_2)
d2m=f2.variables['specific_humidity_2m'][hour_start:hour_end,:,:]

pattern_3= '0_meteo/'+dir_year+'/'+dir_month+'/evaporation_0p01.nc'
filename_3 = dir_+pattern_3
f3= xr.open_dataset(filename_3)
e=f3.variables['evaporation'][hour_start:hour_end,:,:]

pattern_4= '0_meteo/'+dir_year+'/'+dir_month+'/surface_pressure_0p01.nc'
filename_4 = dir_+pattern_4
f4= xr.open_dataset(filename_4)
sp=f4.variables['surface_pressure'][hour_start:hour_end,:,:]

pattern_5= '0_meteo/'+dir_year+'/'+dir_month+'/t2m_0p01.nc'
filename_5 = dir_+pattern_5
f5= xr.open_dataset(filename_5)
t2m=f5.variables['t2m'][hour_start:hour_end,:,:]

pattern_6= '0_meteo/'+dir_year+'/'+dir_month+'/precipitation_0p01.nc'
filename_6 = dir_+pattern_6
f6= xr.open_dataset(filename_6)
tp=f6.variables['precipitation'][hour_start:hour_end,:,:]

pattern_7= '0_meteo/'+dir_year+'/'+dir_month+'/u10_0p01.nc'
filename_7 = dir_+pattern_7
f7= xr.open_dataset(filename_7)
u10=f7.variables['u10'][hour_start:hour_end,:,:]

pattern_8= '0_meteo/'+dir_year+'/'+dir_month+'/v10_0p01.nc'
filename_8 = dir_+pattern_8
f8= xr.open_dataset(filename_8)
v10=f8.variables['v10'][hour_start:hour_end,:,:]

BLH= blh.data
print('blh')
D2M= d2m.data
print('d2m')
E =  e.data
print('e')
SP = sp.data
print('sp')
T2M =t2m.data
print('t2m')
TP = tp.data
print('tp')
U10= u10.data
print('u10')
V10= v10.data
print('v10')
print('BLH shape',np.shape(BLH))
print('BLH type:',type(BLH))

'''
1) AOD
'''

print('1 aod')

pattern_9 = '1_aod/'+dir_year+'/'+dir_month+'/aod_0p01.nc'
filename_9 = dir_ + pattern_9
f9 = xr.open_dataset(filename_9)
AOD=f9.variables['aod'][hour_start:hour_end,:,:]
AOD = AOD.data
print('AOD shape',np.shape(AOD))
print('AOD type',type(AOD))

'''
2) land use cover
'''

print('2 land use conver')

Land_Use_Cover = []
hrs_list = [1+X for X in range(number_of_hours)]  #744 f
for i in range(len(hrs_list)):
    pattern_11 = '2_luc_modis/2023/LUC_0p01_regrid_data.nc'
    filename_11 = dir_ + pattern_11
    f11 = xr.open_dataset(filename_11)
    land_use_cover = f11.variables['luc'][:][0]
    Land_Use_Cover.append(land_use_cover)
Land_Use_Cover = np.array(Land_Use_Cover)
print('LUC shape',np.shape(Land_Use_Cover))
print('LUC type',type(Land_Use_Cover))
'''
3) elevation
'''

print('3 elevation')

Elevation = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    pattern_12 = '3_elevation_srtm/20yearNonChange/elevation_final_regrid_0p01.nc'
    filename_12 = dir_ + pattern_12
    f12 = xr.open_dataset(filename_12)
    elevation = f12.variables['Elevation'][:][0]
    Elevation.append(elevation)
Elevation = np.array(Elevation)
print('Elevation type',type(Elevation))

'''
4) population
'''

print('4 population')

Population = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    pattern_13 = '4_population_landscan/2023/Population_0p01_regrid_data.nc'
    filename_13 = dir_ + pattern_13
    f13 = xr.open_dataset(filename_13)
    population = f13.variables['Population'][:][0]
    Population.append(population)
Population = np.array(Population)
print('population type',type(Population))
'''
5) chem
'''
print('5 chem')
pattern_14 = '5_chem/'+dir_year+'/'+dir_month+'/pm25_0p01.nc'
filename_14 = dir_+ pattern_14
f14 = xr.open_dataset(filename_14)
UFS_AQM_PM25 = f14.variables['pm25'][hour_start:hour_end,:,:]
UFS_AQM_PM25 = UFS_AQM_PM25.data
print('CHEM type',type(UFS_AQM_PM25))
'''
6)anthro emis
'''
print('6 emissions anthropogenic NEMO')
pattern_18 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_bc_0p01_regrid_data.nc'
filename_18 = dir_ + pattern_18
f18 = xr.open_dataset(filename_18)
E_BC_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_BC = f18.variables['E_bc'][:][0]
    E_BC_hourly.append(E_BC)

pattern_19 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_nh3_0p01_regrid_data.nc'
filename_19 = dir_ + pattern_19
f19 = xr.open_dataset(filename_19)
E_NH3_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_NH3 = f19.variables['E_nh3'][:][0]
    E_NH3_hourly.append(E_NH3)

pattern_20 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_nox_0p01_regrid_data.nc'
filename_20 = dir_ + pattern_20
f20 = xr.open_dataset(filename_20)
E_NOX_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_NOX = f20.variables['E_nox'][:][0]
    E_NOX_hourly.append(E_NOX)

pattern_21 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_voc_0p01_regrid_data.nc'
filename_21 = dir_ + pattern_21
f21 = xr.open_dataset(filename_21)
E_VOC_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_VOC = f21.variables['E_voc'][:][0]
    E_VOC_hourly.append(E_VOC)

pattern_22 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_pm25_0p01_regrid_data.nc'
filename_22 = dir_ + pattern_22
f22 = xr.open_dataset(filename_22)
E_PM25_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_PM25 = f22.variables['E_pm25'][:][0]
    E_PM25_hourly.append(E_PM25)

pattern_23 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_so2_0p01_regrid_data.nc'
filename_23 = dir_ + pattern_23
f23 = xr.open_dataset(filename_23)
E_SO2_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_SO2= f23.variables['E_so2'][:][0]
    E_SO2_hourly.append(E_SO2)

E_BC_hourly = np.array(E_BC_hourly)
E_NH3_hourly = np.array(E_NH3_hourly)
E_NOX_hourly = np.array(E_NOX_hourly)
E_VOC_hourly = np.array(E_VOC_hourly)
E_PM25_hourly = np.array(E_PM25_hourly)
E_SO2_hourly = np.array(E_SO2_hourly)
print('Emis so2 shape',np.shape(E_SO2_hourly))
print('emis so2 type',type(E_SO2_hourly))
'''
7) LAT, LON, and distance
'''
print('7 distance')

pattern_24 = '7_Distance/Distance_0p01.nc'
filename_24 = dir_ + pattern_24
f24 = xr.open_dataset(filename_24)

LAT_hourly= f24.variables['LAT'][:]
LON_hourly= f24.variables['LON'][:]
D1= f24.variables['D1'][:]
D2= f24.variables['D2'][:]
D3= f24.variables['D3'][:]
D4= f24.variables['D4'][:]
D5= f24.variables['D5'][:]

LAT_hourly = LAT_hourly.data
LON_hourly = LON_hourly.data
D1 = D1.data
D2 = D2.data
D3 = D3.data
D4 = D4.data
D5 = D5.data
print('D1 shape',np.shape(D1))
print('D1 type',type(D1))
'''
8) Time input
'''

'''
8-1) create datetime
'''
print('8 time')
year = int(dir_year)
month = int(dir_month.replace(dir_year+'_',''))


start_date = dt(year, month, date,start_hour,0,0)
#print('start date',start_date)

date_list = [start_date + timedelta(hours=X) for X in range(number_of_hours)]

hour_list = [] #96
day_list = []  #96
for i in range(len(date_list)):
    date_here = date_list[i]
    hour_here = date_here.hour
    day_here  = date_here.timetuple().tm_yday

    hour_list.append(hour_here)
    day_list.append(day_here)

'''
8-2) create matrix
'''
hour_matrix = []
day_matrix = []
for i in range(len(date_list)):
    hour_matrix_here = np.full((2400,6000),hour_list[i])
    day_matrix_here = np.full((2400,6000),day_list[i])

    hour_matrix.append(hour_matrix_here)
    day_matrix.append(day_matrix_here)

hour_matrix = np.array(hour_matrix)
day_matrix = np.array(day_matrix)

print('hour type',type(hour_matrix))





















































