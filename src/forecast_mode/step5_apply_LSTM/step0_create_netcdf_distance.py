'''
author Beiming.Tang
date: 05/15/2025
'''

import numpy as np
from  netCDF4 import Dataset
from datetime import datetime as dt
from datetime import timedelta
import xarray as xr



'''
0) initialization
'''

dir_= '/data/aqf3/beiming.tang/DAFCOM/data/'
dir_year='2025'
dir_month='2025_04'
#date = 27

number_of_hours= 96
#hour_start = (date-1)*24
#hour_end = (date+3)*24

'''
1) meteorology
'''
print('0 meteo')

pattern_0 = '0_meteo/'+dir_year+'/'+dir_month+'/evaporation_0p01.nc'
filename_0 = dir_+pattern_0
f0= Dataset(filename_0,'r')
LAT_ = f0.variables['lat'][:]
LON_ = f0.variables['lon'][:]

print('finish this section')


'''
7) LAT, LON, and distance
'''
'''
7-1) Lon
'''
print('7 prepare lat and lon')
Lon_matrix = []
for i in range(2400):
    Lon_matrix.append(LON_)
Lon_matrix = np.reshape(Lon_matrix,(2400,6000))

Lon_matrix_hourly = []
for i in range(number_of_hours):
    Lon_matrix_hourly.append(Lon_matrix)
print('LON shape',np.shape(Lon_matrix_hourly))

'''
7-2) Lat
'''
Lat_matrix = []
for i in range(6000):
    Lat_matrix.append(np.transpose(LAT_))
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

D1_ = [] #north west
D2_ = [] #north east
D3_ = [] #south west
D4_ = [] #south east
D5_ = [] #center
for i in range(2400):
    for j in range(6000):
        d1 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-125, 49)
        d2 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-65, 49)
        d3 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-125, 25)
        d4 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-65, 25)
        d5 = haversine(Lon_matrix[i][j],Lat_matrix[i][j],-90, 37)
        D1_.append(d1)
        D2_.append(d2)
        D3_.append(d3)
        D4_.append(d4)
        D5_.append(d5)

D1_ = np.reshape(D1_, (2400,6000))
D2_ = np.reshape(D2_, (2400,6000))
D3_ = np.reshape(D3_, (2400,6000))
D4_ = np.reshape(D4_, (2400,6000))
D5_ = np.reshape(D5_, (2400,6000))

D1_matrix_hourly = []
for i in range(number_of_hours):
    D1_matrix_hourly.append(D1_)
D2_matrix_hourly = []
for i in range(number_of_hours):
    D2_matrix_hourly.append(D2_)
D3_matrix_hourly = []
for i in range(number_of_hours):
    D3_matrix_hourly.append(D3_)
D4_matrix_hourly = []
for i in range(number_of_hours):
    D4_matrix_hourly.append(D4_)
D5_matrix_hourly = []
for i in range(number_of_hours):
    D5_matrix_hourly.append(D5_)
print('Distance shape',np.shape(D1_matrix_hourly))







'''
SECTION 2. write in netcdf
'''
ft = Dataset('Distance_0p01.nc','w',format='NETCDF4')

time = ft.createDimension('time',None)
lat  = ft.createDimension('lat',2400)
lon  = ft.createDimension('lon',6000)

LAT = ft.createVariable('LAT','f4',('time','lat','lon'))
LAT.units = ''
LAT.description = ''

LON = ft.createVariable('LON','f4',('time','lat','lon'))
LON.units = ''
LON.description = ''

D1 = ft.createVariable('D1','f4',('time','lat','lon'))
D1.units = ''
D1.description = ''

D2 = ft.createVariable('D2','f4',('time','lat','lon'))
D2.units = ''
D2.description = ''

D3 = ft.createVariable('D3','f4',('time','lat','lon'))
D3.units = ''
D3.description = ''

D4 = ft.createVariable('D4','f4',('time','lat','lon'))
D4.units = ''
D4.description = ''

D5 = ft.createVariable('D5','f4',('time','lat','lon'))
D5.units = ''
D5.description = ''

'''
write in data
'''
LAT[:,:,:]=Lat_matrix_hourly 
LON[:,:,:]=Lon_matrix_hourly 
D1[:,:,:]= D1_matrix_hourly
D2[:,:,:]= D2_matrix_hourly
D3[:,:,:]= D3_matrix_hourly
D4[:,:,:]= D4_matrix_hourly
D5[:,:,:]= D5_matrix_hourly

ft.close()


print('finsh section2 distance prepare')









