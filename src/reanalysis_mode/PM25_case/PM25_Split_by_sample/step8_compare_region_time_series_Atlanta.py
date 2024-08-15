#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:46:06 2022

@author: btang1
"""
import numpy as np
import os
from scipy import spatial
import fnmatch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import codecs
import datetime
from datetime import datetime as dt
# import h5py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import scipy 

'''
1-1) load OBS Data within domain of interest
'''
def GetData(ll_lat,ur_lat,ll_lon,ur_lon):
    file_loc = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/code_CONUS_US/PM25_split_by_sample_MCMAQ_EGMU_AOD/with_AOD_result/'
    filename = 'CONUS_US_interpolated_Nov2018_withAOD'
    obs_df = pd.read_excel(file_loc+filename+'.xls')
    pd.set_option('display.max_columns',None)  
        
    LOC_NUMBER = list(obs_df['Location_number'])
    EPA_PM25 = list(obs_df['EPA_OBS_PM25'])
    LAT = list(obs_df['LAT'])
    LON = list(obs_df['LON'])
    TIME = list(obs_df['Julian_day'])
    # TIME = list(obs_df['Time_CDT'])
     
    LOC_number_domain = []
    EPA_pm25_domain = []
    LAT_domain = []
    LON_domain = []
    TIME_domain = []
    for i in range(len(LOC_NUMBER)):
        if (LAT[i] - ll_lat)*(LAT[i] - ur_lat) <0:
            if (LON[i]-ll_lon)*(LON[i]-ur_lon) <0:
                LOC_number_domain.append(LOC_NUMBER[i])
                EPA_pm25_domain.append(EPA_PM25[i])
                LAT_domain.append(LAT[i])
                LON_domain.append(LON[i])
                
                time_here_old = int(np.around(TIME[i],0))
                time_here_new = dt.strptime('18'+str(time_here_old),'%y%j').date()
                TIME_domain.append(time_here_new)
         
    return  EPA_pm25_domain,LOC_number_domain,LAT_domain, LON_domain,TIME_domain


# EPA_PM25_domain,LOC_NUMBER_domain,LAT_domain, LON_domain,TIME_domain= GetData(33.7,34.2,-118.6,-117.8) #this is LA
# EPA_PM25_domain,LOC_NUMBER_domain,LAT_domain, LON_domain,TIME_domain= GetData(39.65,39.80,-105.05,-104.9) #this is Denver
# EPA_PM25_domain,LOC_NUMBER_domain,LAT_domain, LON_domain,TIME_domain= GetData(37.3,38.3,-122.7,-121.9) #this is Bay Area
# EPA_PM25_domain,LOC_NUMBER_domain,LAT_domain, LON_domain,TIME_domain= GetData(40.6,40.9,-74.1,-73.7) #this is nyc
EPA_PM25_domain,LOC_NUMBER_domain,LAT_domain, LON_domain,TIME_domain= GetData(33.45,34.05,-84.68,-84.08) #this is atlanta

    
print(len(EPA_PM25_domain))


              
'''
1-2) get uniques stations within domain of interest
'''
Lon_bystation = []
Lat_bystation = []
Location_number_bystation = []
Time_bystation = []
for i in range(1,len(LOC_NUMBER_domain)):
    if LOC_NUMBER_domain[i] != LOC_NUMBER_domain[i-1]:
        Lon_bystation.append(LON_domain[i-1])
        Lat_bystation.append(LAT_domain[i-1])
        Location_number_bystation.append(LOC_NUMBER_domain[i-1])
        Time_bystation.append(TIME_domain[i-1])
        
if Lon_bystation[-1] != LON_domain[-1]: 
    Lon_bystation.append(LON_domain[-1])
    Lat_bystation.append(LAT_domain[-1])
    Location_number_bystation.append(LOC_NUMBER_domain[-1])
    Time_bystation.append(TIME_domain[-1])
    
print(len(Lon_bystation))


'''
1-3) re-arrange obs data in step1 to be (site,time) dimension
'''
EPA_PM25_formal = []
Time_formal = []
for i in range(len(Location_number_bystation )):
    EPA_PM25_formal_this_site = []
    Time_formal_this_site = []
    
    for j in range(len(LOC_NUMBER_domain)):
        if LOC_NUMBER_domain[j] == Location_number_bystation[i]:
            EPA_PM25_formal_this_site.append(EPA_PM25_domain[j])
            Time_formal_this_site.append(TIME_domain[j])
            
    EPA_PM25_formal.append(EPA_PM25_formal_this_site)
    Time_formal.append(Time_formal_this_site)
    
        
PM25_OBS_site1 = EPA_PM25_formal[0]
PM25_OBS_site2 = EPA_PM25_formal[1]
PM25_OBS_site3 = EPA_PM25_formal[2]
PM25_OBS_site4 = EPA_PM25_formal[3]
PM25_OBS_site5 = EPA_PM25_formal[4]


TIME_OBS_site1 = Time_formal[0]
TIME_OBS_site2 = Time_formal[1]
TIME_OBS_site3 = Time_formal[2]
TIME_OBS_site4 = Time_formal[3]
TIME_OBS_site5 = Time_formal[4]


'''
2-1) read in ML prediction data
'''

dir_ml = '/Volumes/Ext_Disk_1/2_GMU_projects/1_GMU_20year_run/CONUS_US/code_CONUS_US/PM25_split_by_sample_MCMAQ_EGMU_AOD/with_AOD_result/'
pattern_ml = 'CONUS_US_surfacePM25_0p01_regird.nc'
filename_ml = dir_ml + pattern_ml
fml = Dataset(filename_ml,'r')
print(list(fml.variables))
LAT_ML = fml.variables['latitude'][:]
LON_ML = fml.variables['longitude'][:]
PM25_ML = fml.variables['Surface pm25'][:] #(30,2500,3500)


def GetML(lon_input,lat_input):
    
    lat_obs = lat_input
    lon_obs = lon_input
    
    lat_cams  = LAT_ML #2500
    lon_cams  = LON_ML #3500
    variable_cams = PM25_ML  # (30,2500,3500)                                       
    
    output_list = []  #this is time list
    
    c= 'no value'
    for i in range(1,2400):            #600
        if (lat_cams[i-1]-lat_obs)*(lat_cams[i]-lat_obs) < 0: 
            
            for j in range(1,6000):   #400         
                if (lon_cams[j-1]-lon_obs)*(lon_cams[j]-lon_obs) < 0:     
                    
                    x1 = np.abs((lon_cams[j-1]-lon_obs)/(lon_cams[j-1]-lon_cams[j]))
                    x2 = np.abs((lon_cams[j]-lon_obs)/(lon_cams[j-1]-lon_cams[j]))
                    
                    y1 = np.abs((lat_cams[i-1]-lat_obs)/(lat_cams[i-1]-lat_cams[i]))
                    y2 = np.abs((lat_cams[i]-lat_obs)/(lat_cams[i-1]-lat_cams[i]))
                    
                    for k in range(30):  #2018-Nov HAVE 30 DAYS
                        a = x2 * variable_cams[k][i-1][j-1] + x1 * variable_cams[k][i-1][j]
                        b = x2 * variable_cams[k][i][j-1] + x1 * variable_cams[k][i][j]
                        c = y1 * b + y2 * a  
                        output_list.append(c)
                        
                    break
            break            
    return output_list

'''
2-1) get ML time series
'''
PM25_ML_site1 = GetML(Lon_bystation[0],Lat_bystation[0])
PM25_ML_site2 = GetML(Lon_bystation[1],Lat_bystation[1])
PM25_ML_site3 = GetML(Lon_bystation[2],Lat_bystation[2])
PM25_ML_site4 = GetML(Lon_bystation[3],Lat_bystation[3])
PM25_ML_site5 = GetML(Lon_bystation[4],Lat_bystation[4])


base_time_ml = dt(2018,11,1,0,0,0)
TIME_ML_all_site = [base_time_ml + datetime.timedelta(days=x) for x in range(30)]
'''
3) Plot compare ML vs. OBS
'''
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%m/%d')
plt.rc('ytick', labelsize=20)
plt.rc('xtick', labelsize=20)

fig = plt.subplots(1,1,figsize = (18,18))
ax1 = plt.subplot(5,1,1)
ax1.xaxis.set_major_formatter(myFmt)
plt.plot(TIME_OBS_site1,PM25_OBS_site1,'k*',label = 'EPA OBS @ site '+str(Location_number_bystation[0]))
plt.plot(TIME_ML_all_site,PM25_ML_site1,'r-',label = 'ML prediction')
plt.tick_params(labelbottom = False, bottom = False)
plt.xlim(dt(2018,11,1,0,0,0),dt(2018,12,1,0,0,0))
plt.ylim(0,40)
plt.ylabel('$PM_{2.5}$ \n (\u03bcg/$m^3$)',fontsize = 20)
plt.legend(loc = 'upper right',fontsize = 20)

ax2 = plt.subplot(5,1,2)
ax2.xaxis.set_major_formatter(myFmt)
plt.plot(TIME_OBS_site2,PM25_OBS_site2,'k*',label = 'EPA OBS @ site '+str(Location_number_bystation[1]))
plt.plot(TIME_ML_all_site,PM25_ML_site2,'r-',label = 'ML prediction')
plt.tick_params(labelbottom = False, bottom = False)
plt.xlim(dt(2018,11,1,0,0,0),dt(2018,12,1,0,0,0))
plt.ylim(0,40)
plt.ylabel('$PM_{2.5}$ \n (\u03bcg/$m^3$)',fontsize = 20)
plt.legend(loc = 'upper right',fontsize = 20)

ax3 = plt.subplot(5,1,3)
ax3.xaxis.set_major_formatter(myFmt)
plt.plot(TIME_OBS_site3,PM25_OBS_site3,'k*',label = 'EPA OBS @ site '+str(Location_number_bystation[2]))
plt.plot(TIME_ML_all_site,PM25_ML_site3,'r-',label = 'ML prediction')
plt.tick_params(labelbottom = False, bottom = False)
plt.xlim(dt(2018,11,1,0,0,0),dt(2018,12,1,0,0,0))
plt.ylim(0,40)
plt.ylabel('$PM_{2.5}$ \n (\u03bcg/$m^3$)',fontsize = 20)
plt.legend(loc = 'upper right',fontsize = 20)

ax4 = plt.subplot(5,1,4)
ax4.xaxis.set_major_formatter(myFmt)
plt.plot(TIME_OBS_site4,PM25_OBS_site4,'k*',label = 'EPA OBS @ site '+str(Location_number_bystation[3]))
plt.plot(TIME_ML_all_site,PM25_ML_site4,'r-',label = 'ML prediction')
plt.tick_params(labelbottom = False, bottom = False)
plt.xlim(dt(2018,11,1,0,0,0),dt(2018,12,1,0,0,0))
plt.ylim(0,40)
plt.ylabel('$PM_{2.5}$ \n (\u03bcg/$m^3$)',fontsize = 20)
plt.legend(loc = 'upper right',fontsize = 20)

ax5 = plt.subplot(5,1,5)
ax5.xaxis.set_major_formatter(myFmt)
plt.plot(TIME_OBS_site5,PM25_OBS_site5,'k*',label = 'EPA OBS @ site '+str(Location_number_bystation[4]))
plt.plot(TIME_ML_all_site,PM25_ML_site5,'r-',label = 'ML prediction')
plt.tick_params(labelbottom = True, bottom = True)
plt.xlim(dt(2018,11,1,0,0,0),dt(2018,12,1,0,0,0))
plt.ylim(0,40)
plt.ylabel('$PM_{2.5}$ \n (\u03bcg/$m^3$)',fontsize = 20)
plt.legend(loc = 'upper right',fontsize = 20)
plt.savefig('Atlanta_compare_5sites_withAOD',dpi = 600)