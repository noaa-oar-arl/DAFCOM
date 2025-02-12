#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:49:54 2022

@author: btang1
"""


import numpy as np
#import os
#from scipy import spatial
#import fnmatch
#from netCDF4 import Dataset
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.basemap import Basemap
#import codecs
#import datetime
from datetime import datetime as dt
import pandas as pd
#import statistics as stat
#from math import nan, isnan



'''
1) import GS PM25 observation data
'''

file_loc = '/groups/ESS3/btang6/DATA/gs_data/US/EPA_US/2023/' 
filename = 'daily_pm25_2023'
obs_df = pd.read_csv(file_loc+filename+'.csv')
pd.set_option('display.max_columns',None)  

time_raw = list(obs_df['Date Local'])
Variable_obs_raw = list(obs_df['Arithmetic Mean'])
latitude_raw = list(obs_df['Latitude'])
longitude_raw = list(obs_df['Longitude'])
county_code_raw = list(obs_df['County Code'])
site_code_raw = list(obs_df['Site Num'])
poc_raw = list(obs_df['POC'])


'''
2) get daily averaged in target time and domain, kick out negative values
but this total data for each site may have multiple same time with different measurment from different sensors,
or same time different around digit precisiosn, need to go to step 4 for further treatment
'''

Time_new_list = []
PM25_OBS_list = []
LAT_OBS_list = []
LON_OBS_list = []
# COUNTY_CODE_OBS_list = []
# SITE_CODE_OBS_list = []
LOCATION_NUMBER_OBS_List = []
# POC_OBS_List = []
for i in range(len(time_raw)):
    time_new = dt.strptime(str(time_raw[i]),'%Y-%m-%d')
    location_numer = str(county_code_raw[i])+'_'+str(site_code_raw[i])
    if time_new >= dt(2023,6,1,0,0,0) and time_new < dt(2023,7,1,0,0,0):
        if (latitude_raw[i]-25)*(latitude_raw[i]-49) <= 0:
            if (longitude_raw[i]-(-125))*(longitude_raw[i]-(-65))<= 0:
                if Variable_obs_raw[i] >= 0:
                    Time_new_list.append(time_new)
                    PM25_OBS_list.append(Variable_obs_raw[i])
                    LAT_OBS_list.append(latitude_raw[i])
                    LON_OBS_list.append(longitude_raw[i])
                    # COUNTY_CODE_OBS_list.append(county_code_raw[i])
                    # SITE_CODE_OBS_list.append(site_code_raw[i])
                    LOCATION_NUMBER_OBS_List.append(location_numer)
                    # POC_OBS_List.append(poc_raw[i])


print('total daily data = ',len(PM25_OBS_list))
# print(np.min(PM25_OBS_list),np.max(PM25_OBS_list))
# print(np.min(LAT_OBS_list),np.max(LAT_OBS_list))
# print(np.min(LON_OBS_list),np.max(LON_OBS_list))
# print(np.min(Time_new_list),np.max(Time_new_list))

'''
3) print # of stations
'''
Count_loc_number = []
Count_loc_number.append(LOCATION_NUMBER_OBS_List[0])
for i in range(1,len(LOCATION_NUMBER_OBS_List)):
    if LOCATION_NUMBER_OBS_List[i-1] != LOCATION_NUMBER_OBS_List[i]:
        Count_loc_number.append(LOCATION_NUMBER_OBS_List[i])
print('CONUS US number of locations = ',len(Count_loc_number))

LAT_count = []
LON_count = []
for i in range(len(Count_loc_number)):
    # print(i)
    for j in range(len(LOCATION_NUMBER_OBS_List )):
        if Count_loc_number[i] == LOCATION_NUMBER_OBS_List[j]:
            LAT_count.append(LAT_OBS_list[j])
            LON_count.append(LON_OBS_list[j])
            break
'''
4) average for same site but different POC
'''
'''
4-1)re-arrange dimension into (site,#values)
'''
Time_count_list = []
PM_count_list = []
# POC_count_list = []
for i in range(len(Count_loc_number)):
    time_count = []
    pm_count = []
    # poc_count = []
    
    for j in range(len(LOCATION_NUMBER_OBS_List )):
        if LOCATION_NUMBER_OBS_List[j] == Count_loc_number[i]:
            time_count.append(Time_new_list[j])
            pm_count.append(PM25_OBS_list[j])
            # poc_count.append(POC_OBS_List[j])
            
    Time_count_list.append(time_count)
    PM_count_list.append(pm_count)
    # POC_count_list.append(poc_count)
            
# print(len(Time_count_list))                   
                    
'''
4-2)get uniques time per site
'''
Time_Unique_list = [] #452
for i in range(len(Count_loc_number)): #452
    time_unique = []
    for j in range(len(Time_count_list[i])): #time per site
        if Time_count_list[i][j] not in time_unique:
            # print(i,j)
            time_unique.append(Time_count_list[i][j])
    Time_Unique_list.append(time_unique)

'''
4-3) get time averaged pm2.5 per site
'''
PM_Unique_list = [] #452
for i in range(len(Count_loc_number)):
    
    pm_unique_per_site = []    
    for j in range(len(Time_Unique_list[i])):  #this is A list   
        time_unique_here = Time_Unique_list[i][j]
        index_unique_here = 0
        pm_unique_here = 0
        
        for k in range(len(Time_count_list[i])): #this is B list
            time_here = Time_count_list[i][k]
            
            if time_here == time_unique_here:
                index_unique_here += 1
                pm_unique_here  += PM_count_list[i][k]
                
        pm_unique_final = np.around(pm_unique_here/index_unique_here,3)               
        pm_unique_per_site.append(pm_unique_final)    
    PM_Unique_list.append(pm_unique_per_site)
       
            
                       
'''
5) prepare final output list
'''
LOC_NUMBER_OBS_FINAL = []
TIME_OBS_FINAL = []
PM_OBS_FINAL = []
LAT_OBS_FINAL = []
LON_OBS_FINAL = []

for i in range(len(Time_Unique_list)):
    for j in range(len(Time_Unique_list[i])):
        LOC_NUMBER_OBS_FINAL.append(Count_loc_number[i])
        LAT_OBS_FINAL.append(LAT_count[i])
        LON_OBS_FINAL.append(LON_count[i])
        TIME_OBS_FINAL.append(Time_Unique_list[i][j]) 
        PM_OBS_FINAL.append(PM_Unique_list[i][j])
                             

print(len(LOC_NUMBER_OBS_FINAL))
print(len(TIME_OBS_FINAL))
print(len(PM_OBS_FINAL))
print(len(LAT_OBS_FINAL))
print(len(LON_OBS_FINAL))



'''
6)plot the location of above stations
'''


# m = Basemap(projection='cyl', resolution='l',llcrnrlat=25, urcrnrlat = 49,llcrnrlon=-125, urcrnrlon = -65) 
# fig = plt.subplots(1,1,figsize = (18,10)) 


# m.drawparallels(np.arange(25, 49, 10), labels=[1, 0, 0, 0],fontsize = 30)
# m.drawmeridians(np.arange(-125, -65, 10), labels=[0, 0, 0, 1],fontsize = 30)

# plt.plot(np.array(LON_count), np.array(LAT_count),'r*',label = 'EPA no2 sites')
# plt.legend(loc='lower left',fontsize=30)

# m.drawcoastlines(linewidth = 3)
# m.drawcountries(linewidth = 3)
# m.drawstates(linewidth = 1)


# plt.savefig('CONUS_US_map_no2_sites.png', dpi = 300)

    


















