'''
author:Beiming Tang
date: 05/02/2025
'''
import numpy as np
from netCDF4 import Dataset
from datetime import datetime as dt
from datetime import timedelta

'''
0) initialization
'''

dir_='/data/aqf3/beiming.tang/DAFCOM/data/'
dir_year = '2025'    #CHANGE
dir_month = '2025_04'#CHANGE
date= 1              #CHANGE

number_of_hours= 24
hour_start = (date-1)*24
hour_end = date*24
print('this is '+dir_month+' '+str(date)+' input data.')

'''
0) meteorology
'''
print('0 meteo')
pattern_0 = '0_meteo/'+dir_year+'/'+dir_month+'/evaporation_0p01.nc'
filename_0 = dir_+pattern_0
f0= Dataset(filename_0,'r')
LAT = f0.variables['lat'][:]
LON = f0.variables['lon'][:]

pattern_1= '0_meteo/'+dir_year+'/'+dir_month+'/pblh_0p01.nc'
filename_1 = dir_+pattern_1
f1=Dataset(filename_1,'r')
blh=f1.variables['pblh'][hour_start:hour_end,:,:]

pattern_2= '0_meteo/'+dir_year+'/'+dir_month+'/specific_humidity_2m_0p01.nc'
filename_2 = dir_+pattern_2
f2=Dataset(filename_2,'r')
d2m=f2.variables['specific_humidity_2m'][hour_start:hour_end,:,:]

pattern_3= '0_meteo/'+dir_year+'/'+dir_month+'/evaporation_0p01.nc'
filename_3 = dir_+pattern_3
f3=Dataset(filename_3,'r')
e=f3.variables['evaporation'][hour_start:hour_end,:,:]

pattern_4= '0_meteo/'+dir_year+'/'+dir_month+'/surface_pressure_0p01.nc'
filename_4 = dir_+pattern_4
f4=Dataset(filename_4,'r')
sp=f4.variables['surface_pressure'][hour_start:hour_end,:,:]

pattern_5= '0_meteo/'+dir_year+'/'+dir_month+'/t2m_0p01.nc'
filename_5 = dir_+pattern_5
f5=Dataset(filename_5,'r')
t2m=f5.variables['t2m'][hour_start:hour_end,:,:]

pattern_6= '0_meteo/'+dir_year+'/'+dir_month+'/precipitation_0p01.nc'
filename_6 = dir_+pattern_6
f6=Dataset(filename_6,'r')
tp=f6.variables['precipitation'][hour_start:hour_end,:,:]

pattern_7= '0_meteo/'+dir_year+'/'+dir_month+'/u10_0p01.nc'
filename_7 = dir_+pattern_7
f7=Dataset(filename_7,'r')
u10=f7.variables['u10'][hour_start:hour_end,:,:]

pattern_8= '0_meteo/'+dir_year+'/'+dir_month+'/v10_0p01.nc'
filename_8 = dir_+pattern_8
f8=Dataset(filename_8,'r')
v10=f8.variables['v10'][hour_start:hour_end,:,:]

BLH=blh
D2M=d2m
E = e
SP = sp
T2M = t2m
TP = tp
U10=u10
V10=v10
#print('BLH',np.shape(BLH))
'''
1) AOD
'''
print('1 aod')
pattern_9 = '1_aod/'+dir_year+'/'+dir_month+'/aod_0p01.nc'
filename_9 = dir_ + pattern_9
f9 = Dataset(filename_9,'r')
AOD=f9.variables['aod'][hour_start:hour_end,:,:]
#print('AOD',np.shape(AOD))
'''
2) land use cover
'''
print('2 land use conver')
Land_Use_Cover = []
hrs_list = [1+X for X in range(number_of_hours)]  #744 f
for i in range(len(hrs_list)):
    pattern_11 = '2_luc_modis/2023/LUC_0p01_regrid_data.nc'
    filename_11 = dir_ + pattern_11
    f11 = Dataset(filename_11,'r')
    land_use_cover = f11.variables['luc'][:][0]
    Land_Use_Cover.append(land_use_cover)
#print('LUC',np.shape(Land_Use_Cover))
'''
3) elevation
'''
print('3 elevation')
Elevation = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    pattern_12 = '3_elevation_srtm/20yearNonChange/elevation_final_regrid_0p01.nc'
    filename_12 = dir_ + pattern_12
    f12 = Dataset(filename_12,'r')
    elevation = f12.variables['Elevation'][:][0]
    Elevation.append(elevation)
#print('Elevation',np.shape(Elevation))
'''
4) population
'''
print('4 population')
Population = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    pattern_13 = '4_population_landscan/2023/Population_0p01_regrid_data.nc'
    filename_13 = dir_ + pattern_13
    f13 = Dataset(filename_13,'r')
    population = f13.variables['Population'][:][0]
    Population.append(population)
#print('Population',np.shape(Population))
'''
5) chem
'''
print('5 chem')
pattern_14 = '5_chem/'+dir_year+'/'+dir_month+'/pm25_0p01.nc'
filename_14 = dir_+ pattern_14
f14 = Dataset(filename_14,'r')
UFS_AQM_PM25 = f14.variables['pm25'][hour_start:hour_end,:,:]
#print('Chem',np.shape(UFS_AQM_PM25))
'''
6)anthro emis
'''
print('6 emissions anthropogenic NEMO')
pattern_18 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_bc_0p01_regrid_data.nc'
filename_18 = dir_ + pattern_18
f18 = Dataset(filename_18,'r')
E_BC_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_BC = f18.variables['E_bc'][:][0]
    E_BC_hourly.append(E_BC)

pattern_19 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_nh3_0p01_regrid_data.nc'
filename_19 = dir_ + pattern_19
f19 = Dataset(filename_19,'r')
E_NH3_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_NH3 = f19.variables['E_nh3'][:][0]
    E_NH3_hourly.append(E_NH3)

pattern_20 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_nox_0p01_regrid_data.nc'
filename_20 = dir_ + pattern_20
f20 = Dataset(filename_20,'r')
E_NOX_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_NOX = f20.variables['E_nox'][:][0]
    E_NOX_hourly.append(E_NOX)

pattern_21 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_voc_0p01_regrid_data.nc'
filename_21 = dir_ + pattern_21
f21 = Dataset(filename_21,'r')
E_VOC_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_VOC = f21.variables['E_voc'][:][0]
    E_VOC_hourly.append(E_VOC)

pattern_22 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_pm25_0p01_regrid_data.nc'
filename_22 = dir_ + pattern_22
f22 = Dataset(filename_22,'r')
E_PM25_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_PM25 = f22.variables['E_pm25'][:][0]
    E_PM25_hourly.append(E_PM25)

pattern_23 = '6_emisanthro_nemo/2019/'+dir_month.replace(dir_year,'2019')+'/E_so2_0p01_regrid_data.nc'
filename_23 = dir_ + pattern_23
f23 = Dataset(filename_23,'r')
E_SO2_hourly = []
hrs_list = [1+X for X in range(number_of_hours)]
for i in range(len(hrs_list)):
    E_SO2= f23.variables['E_so2'][:][0]
    E_SO2_hourly.append(E_SO2)
#print('Emis',np.shape(E_SO2_hourly))
'''
7) LAT, LON, and distance
'''
'''
7-1) Lon
'''
print('7 prepare lat and lon')
Lon_matrix = []
for i in range(2400):
    Lon_matrix.append(LON)
Lon_matrix = np.reshape(Lon_matrix,(2400,6000))

Lon_matrix_hourly = []
for i in range(number_of_hours):
    Lon_matrix_hourly.append(Lon_matrix)
#print('LON',np.shape(Lon_matrix_hourly))
'''
7-2) Lat
'''
Lat_matrix = []
for i in range(6000):
    Lat_matrix.append(np.transpose(LAT))
Lat_matrix = np.reshape(Lat_matrix,(6000,2400))
Lat_matrix = np.transpose(Lat_matrix)

Lat_matrix_hourly = []
for i in range(number_of_hours):
    Lat_matrix_hourly.append(Lat_matrix)

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

D1_matrix_hourly = []
for i in range(number_of_hours):
    D1_matrix_hourly.append(D1)
D2_matrix_hourly = []
for i in range(number_of_hours):
    D2_matrix_hourly.append(D2)
D3_matrix_hourly = []
for i in range(number_of_hours):
    D3_matrix_hourly.append(D3)
D4_matrix_hourly = []
for i in range(number_of_hours):
    D4_matrix_hourly.append(D4)
D5_matrix_hourly = []
for i in range(number_of_hours):
    D5_matrix_hourly.append(D5)
#print('Distance',np.shape(D1_matrix_hourly))
'''
8)date
'''
print('8 date')

year=int(dir_year)
month = int(dir_month.replace(dir_year,'').replace('_',''))
start_date = dt(year,month,date,12,0,0)
#date_list = [start_date + timedelta(hours= X) for X in range(number_of_hours)]

#def GetDateMatrix(date_input):
#    date_matrix = []
#    for i in range(14400000):
#        date_matrix.append(date_input)
#    date_matrix = np.reshape(date_matrix,(2400,6000))
#    return date_matrix
#
#date_matrix_hourly = []
#for i in range(number_of_hours): 
#    date_here = date_list[i]
#    date_matrix_here = GetDateMatrix(date_here)
#    date_matrix_hourly.append(date_matrix_here)
#print('date',np.shape(date_matrix_hourly))




























