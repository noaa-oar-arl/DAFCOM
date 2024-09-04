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
1) load OBS Data
'''

def GetData(filename):
    file_loc = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/code/NO2_splitbysample_MUfs_ENemo_TEMPO/'
    obs_df = pd.read_excel(file_loc+filename+'.xls')
    pd.set_option('display.max_columns',None)  
        
    LOC_NUMBER = list(obs_df['Location_number'])
    EPA_PM25 = list(obs_df['EPA_OBS_NO2'])
    LAT = list(obs_df['LAT'])
    LON = list(obs_df['LON'])
 
    
    return  EPA_PM25,LOC_NUMBER,LAT, LON


EPA_PM25,LOC_NUMBER,LAT, LON,= GetData('CONUS_US_interpolated_Aug2023_NO2')




Y =  EPA_PM25
              
'''
2) process OBS by station
'''
'''
2-1) pm2.5 OBS by station
'''

  
Y_diff = Y   #4885 points = station * (31-1)days




Loc_number_bystation = [] #580
sum_days_bystation = 0
sum_y_diff_bystation = 0
Y_diff_bystation = []    #580
for i in range(1,len(LOC_NUMBER)):
    
    if LOC_NUMBER[i] != LOC_NUMBER[i-1]:
        Loc_number_bystation.append(LOC_NUMBER[i-1]) 
        sum_y_diff_bystation += Y_diff[i-1]
        sum_days_bystation += 1
        Y_diff_bystation.append(sum_y_diff_bystation/sum_days_bystation)
        sum_days_bystation = 0
        sum_y_diff_bystation = 0
       
       
       
    else:
        sum_y_diff_bystation += Y_diff[i-1]
        sum_days_bystation += 1

        
if Loc_number_bystation[-1] != LOC_NUMBER[-1]:
    Loc_number_bystation.append(LOC_NUMBER[-1])
    Y_diff_bystation.append(sum_y_diff_bystation/sum_days_bystation)
    
print(len(Loc_number_bystation))
print(len(Y_diff_bystation))



'''
2-2) get lat and lon for each stations
'''


Lon_bystation = []
Lat_bystation = []
for i in range(1,len(LOC_NUMBER)):
    if LOC_NUMBER[i] != LOC_NUMBER[i-1]:
        Lon_bystation.append(LON[i-1])
        Lat_bystation.append(LAT[i-1])
        
if Lon_bystation[-1] != LON[-1]: 
    Lon_bystation.append(LON[-1])
    Lat_bystation.append(LAT[-1])
    
print(len(Lon_bystation))

'''
3) prepare ML prediction PM2.5, averaged in month
'''
dir_1 = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/data/0_meteo/step2_0p01_fast/'
pattern_0 = 'meteo_blh_0p01.nc'
filename_0 = dir_1 + pattern_0
f0 = Dataset(filename_0,'r')
LAT = f0.variables['lat'][:] #2500
LON = f0.variables['lon'][:] #3500

dir_2 = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/code/NO2_splitbysample_MUfs_ENemo_TEMPO/'
pattern_1 = 'CONUS_US_surfaceNO2_0p01_regird.nc'
filename_1 = dir_2 + pattern_1
f1 = Dataset(filename_1,'r')
# print(list(f1.variables))
Y_hat_month_daily = f1.variables['Surface no2'][:] #(30,2500,3500)

'''
3-1) make averages
'''
Y_hat_month = np.mean(Y_hat_month_daily ,axis = 0)

            
'''
4) plot result into a map
'''
'''
4-1) CONUS domain
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='l',llcrnrlat=25, urcrnrlat = 49,llcrnrlon=-125, urcrnrlon = -65) 
fig = plt.subplots(1,1,figsize = (20,10)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 25,cmap = 'jet') 

m.drawcoastlines(linewidth = 1)
m.drawcountries(linewidth = 1)
m.drawstates(linewidth = 1)

                                  
#plot airkorea obs scatter maps
plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='w',s = 50)
cb = plt.colorbar()
plt.clim(0,15)
cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  


#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')       
plt.savefig('OBS_no2_CONUS.png', dpi = 600)
    

'''
4-2) California
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='l',llcrnrlat=31, urcrnrlat = 43,llcrnrlon=-125, urcrnrlon = -114) 
fig = plt.subplots(1,1,figsize = (15,10)) 
m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 15,cmap = 'jet') 

m.drawcoastlines(linewidth = 1)
m.drawcountries(linewidth = 1)
m.drawstates(linewidth = 1)

                                  
#plot airkorea obs scatter maps
plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='w',s = 50)
cb = plt.colorbar()
plt.clim(0,15)
cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  


#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')     
plt.savefig('ML&OBS_no2_California.png', dpi = 600)


'''
4-3) LA
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='l',llcrnrlat=33.7, urcrnrlat = 34.2,llcrnrlon=-118.6, urcrnrlon = -117.8) 
fig = plt.subplots(1,1,figsize = (15,10)) 
m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 15,cmap = 'jet') 

                                  
#plot airkorea obs scatter maps
m.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',s = 300)
# cb = plt.colorbar(location= 'bottom')
cb = plt.colorbar()
plt.clim(0,15)
cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  


#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/10_LA_road/'
m.readshapefile(dir_shapefile+"tl_2018_06037_roads",'la_roads',linewidth = 0.3)
plt.savefig('ML&OBS_no2_LA.png', dpi = 600)

'''
4-4) denver
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='l',llcrnrlat=39.65, urcrnrlat = 39.80,llcrnrlon=-105.05, urcrnrlon = -104.9) 
fig = plt.subplots(1,1,figsize = (15,10)) 
m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 12,cmap = 'jet') 
                                
#plot airkorea obs scatter maps
plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',linewidths=3, edgecolors='k',s = 300)
cb = plt.colorbar()
plt.clim(0,12)
cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  

#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')    
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/7_Denver_road/'
m.readshapefile(dir_shapefile+"tl_2019_08031_roads",'denver_roads',linewidth = 0.3)  
plt.savefig('ML&OBS_no2__Denver.png', dpi = 600)


# '''
# 4-5) Bay Area
# '''
# # #plot monthly averge map 
# m = Basemap(projection='cyl', resolution='l',llcrnrlat=37.3, urcrnrlat = 38.3,llcrnrlon=-122.7, urcrnrlon = -121.9) 
# fig = plt.subplots(1,1,figsize = (15,10)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin = 0, vmax = 40,cmap = 'jet') 
                                
# #plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',s = 300)
# cb = plt.colorbar()
# plt.clim(0,40)
# cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)  

# #add shape file
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/BayArea_county/'
# m.readshapefile(dir_shapefile+"geo_export_458ef7a3-f2ca-47d8-aea8-de2ea3abf661",'US_states')    

# plt.savefig('ML&OBS_no2_BayArea.png', dpi = 600)

# '''
# 4-1) NYC
# '''
# # #plot monthly averge map 
# m = Basemap(projection='cyl', resolution='h',llcrnrlat=40.6, urcrnrlat = 40.9,llcrnrlon=-74.1, urcrnrlon = -73.7) 
# fig = plt.subplots(1,1,figsize = (18,10)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin =0 , vmax = 40,cmap = 'jet') 
                                
# #plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',
#             edgecolors='k',linewidth=3,s = 100)
# cb = plt.colorbar()
# plt.clim(0,40)
# cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)  

# m.drawcoastlines(linewidth = 1)
# m.drawcountries(linewidth = 1)
# m.drawstates(linewidth = 1)
# #add shape file
# # dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# # m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')    
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/1_NYC_road/'
# m.readshapefile(dir_shapefile+"geo_export_24d583de-a1d3-4667-a4b0-591d6f74cf60",'nyc_road',linewidth=0.1)
# plt.savefig('ML&OBS_no2_NYC.png', dpi = 600)


'''
4-2) DC
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='h',llcrnrlat=38.8, urcrnrlat = 39.0,llcrnrlon=-77.13, urcrnrlon = -76.9) 
fig = plt.subplots(1,1,figsize = (18,10)) 
m.pcolor(LON, LAT, Y_hat_month,vmin =0 , vmax = 12,cmap = 'jet') 
                                
#plot airkorea obs scatter maps
plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',
            edgecolors='k',linewidth=3,s = 300)
cb = plt.colorbar()
plt.clim(0,12)
cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  

m.drawcountries(linewidth = 1)
# m.drawstates(linewidth = 1)
#add shape file
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')    
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/2_DC_road/'
m.readshapefile(dir_shapefile+"Roads",'DC_road',linewidth=0.1)
plt.savefig('ML&OBS_no2_DC.png', dpi = 600)


# '''
# 4-3) Atlanta
# '''
# # #plot monthly averge map 
# m = Basemap(projection='cyl', resolution='h',llcrnrlat=33.45, urcrnrlat = 34.05,llcrnrlon=-84.68, urcrnrlon = -84.08) 
# fig = plt.subplots(1,1,figsize = (18,10)) 
# m.pcolor(LON, LAT, Y_hat_month,vmin =0 , vmax = 40,cmap = 'jet') 
                                
# #plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',
#             edgecolors='k',linewidth=3,s = 300)
# cb = plt.colorbar()
# plt.clim(0,40)
# cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)  

# # m.drawcoastlines(linewidth = 1)
# m.drawcountries(linewidth = 1)
# m.drawstates(linewidth = 1)
# #add shape file
# # dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# # m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')    
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/3_Atlanta_road/'
# m.readshapefile(dir_shapefile+"Major_Roads",'road')
# plt.savefig('ML&OBS_no2_atlanta.png', dpi = 600)


'''
4-4) Alberquque, NM
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='i',llcrnrlat=35.00, urcrnrlat = 35.22,llcrnrlon=-106.78, urcrnrlon = -106.48) # Alberqueuq, NM
fig = plt.subplots(1,1,figsize = (15,10)) 
m.pcolor(LON, LAT, Y_hat_month,vmin =0 , vmax = 12,cmap = 'jet') 

#plot airkorea obs scatter maps
plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',
            edgecolors='k',linewidth=3,s = 300)
cb = plt.colorbar()
plt.clim(0,12)
cb.set_label('NO$_2$ (ppbv)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  

m.drawcountries(linewidth = 1)


#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/Alberquque_NM/'
m.readshapefile(dir_shapefile+"alberquque_streets",'alberquque_streets')      
plt.savefig('ML&OBS_no2_Alberquque.png', dpi = 600)










