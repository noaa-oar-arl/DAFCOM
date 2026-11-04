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

dir_='/data/aqf3/beiming.tang/DAFCOM/data/99_bias_corr_pm25/'
dir_year = '2025'    #CHANGE
dir_month = '2025_05'#CHANGE
date= 1              #CHANGE

number_of_hours= 24
hour_start = (date-1)*24
hour_end = date*24
#print('this is '+dir_month+' '+str(date)+' input data.')

'''
0) bias correction PM25
'''
print('0 bias correction pm25')
pattern_0 = dir_year+'/'+dir_month+'/'+'pm25_bc_0p01.nc'
filename_0 = dir_+pattern_0
f0= Dataset(filename_0,'r')
LAT = f0.variables['lat'][:]
LON = f0.variables['lon'][:]
PM25_BC = f0.variables['pm25_bc'][hour_start:hour_end,:,:]
print('lat',np.max(LAT),np.min(LAT))
print('lon',np.max(LON),np.min(LON))

'''
8)date
'''
print('8 date')

year=int(dir_year)
month = int(dir_month.replace(dir_year,'').replace('_',''))
start_date = dt(year,month,date,12,0,0)





























