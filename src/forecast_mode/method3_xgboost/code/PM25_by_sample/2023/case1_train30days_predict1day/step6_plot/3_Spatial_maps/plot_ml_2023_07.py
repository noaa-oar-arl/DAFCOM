#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:07:08 2022

@author: btang1
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

'''
1) PM UFS
'''
dir_ = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method3_xgboost/code/PM25_by_sample/2023/case1_train30days_predict1day/step6_plot/3_Spatial_maps/'

pattern_0 = 'XGB_PM25_202307.nc'
filename_0 = dir_ + pattern_0
f0 = Dataset(filename_0,'r')
LAT_ml = f0.variables['lat'][:] # (600)
LON_ml = f0.variables['lon'][:] #(400)
NO2 = f0.variables['pm25'][:]

v_input = NO2 #CHANGE
V_month_avr = v_input[0]   #2020-03-01 is 60
for i in range(1,len(v_input )):
    V_month_avr += v_input[i]
    
V_month_avr = V_month_avr /len(v_input )

#get rid of ocean values
V_month_final = np.zeros((len(V_month_avr),len(V_month_avr[0])))
bm = Basemap()
for i in range(len(V_month_avr)):
    for j in range(len(V_month_avr[0])):
        lon_local = LON_ml[j]
        lat_local = LAT_ml[i]
        # print(lon_local,lat_local)
        # print(bm.is_land(lon_local,lat_local))
        if bm.is_land(lon_local,lat_local) == True:
            print(i,j)
            V_month_final [i][j] = V_month_avr[i][j]
        else:
            V_month_final [i][j] = np.nan
            
            
            
'''
2) get OBS 
'''

def GetData(filename):
    file_loc = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method3_xgboost/code/PM25_by_sample/2023/case1_train30days_predict1day/step6_plot/Visualize_UFS/'
    obs_df = pd.read_excel(file_loc+filename+'.xls')
    pd.set_option('display.max_columns',None)  
        
    LOC_NUMBER = list(obs_df['Location_number'])
    EPA_PM25 = list(obs_df['EPA_OBS_PM25'])
    LAT = list(obs_df['LAT'])
    LON = list(obs_df['LON'])
 
    
    return  EPA_PM25,LOC_NUMBER,LAT, LON

EPA_PM25,LOC_NUMBER,LAT, LON,= GetData('CONUS_US_interpolated_2023_183_212_PM25') #THIS IS JULY, 2023
Y =  EPA_PM25
              

#process obs by station
  
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

#get obs for each station
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
3) PLOT
'''
    
'''
3-3) LA
'''
# # #plot monthly averge map 
# m = Basemap(projection='cyl', resolution='l',llcrnrlat=33.7, urcrnrlat = 34.2,llcrnrlon=-118.6, urcrnrlon = -117.8) 
# fig = plt.subplots(1,1,figsize = (15,10)) 
# m.pcolor(LON_ml, LAT_ml, V_month_avr,vmin = 0, vmax = 25,cmap = 'jet') 

                                  
# #plot airkorea obs scatter maps
# m.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',linewidth=3,s = 300)
# # cb = plt.colorbar(location= 'bottom')
# cb = plt.colorbar()
# plt.clim(0,25)
# cb.set_label('PM$_{2.5}$ (\u03bcg/m$^3$)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)  


# #add shape file
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/10_LA_road/'
# m.readshapefile(dir_shapefile+"tl_2018_06037_roads",'la_roads',linewidth = 0.3)
# plt.savefig('UFS_PM25_LA.png', dpi = 600) 
    
    
'''
3-4) Alberquque, NM
'''
# # #plot monthly averge map 
# m = Basemap(projection='cyl', resolution='i',llcrnrlat=35.00, urcrnrlat = 35.22,llcrnrlon=-106.78, urcrnrlon = -106.48) # Alberqueuq, NM
# fig = plt.subplots(1,1,figsize = (15,10)) 
# m.pcolor(LON_ml, LAT_ml,V_month_avr,vmin = 0, vmax = 12,cmap = 'jet') 

# # m.drawcoastlines(linewidth = 2)
# m.drawcountries(linewidth = 2)
# m.drawstates(linewidth = 2)

                                  
# #plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',linewidth=3,s = 300)
# cb = plt.colorbar()
# plt.clim(0,12)
# cb.set_label('PM$_{2.5}$ (\u03bcg/m$^3$)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)  


# #add shape file
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/Alberquque_NM/'
# m.readshapefile(dir_shapefile+"alberquque_streets",'alberquque_streets')      
# plt.savefig('UFS_PM25_Alberquque.png', dpi = 600)


'''
3-1) Atlanta
'''
# m = Basemap(projection='cyl', resolution='l',llcrnrlat=33.45, urcrnrlat = 34.05,llcrnrlon=-84.68, urcrnrlon = -84.08)
# fig = plt.subplots(1,1,figsize = (18,10))
# m.pcolor(LON_ml, LAT_ml, V_month_avr,vmin =12 , vmax = 16,cmap = 'jet')

# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',linewidth=3,s = 300)
# cb = plt.colorbar()
# plt.clim(12,16)
# cb.set_label('PM$_{2.5}$ (\u03bcg/m$^3$)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)

# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/3_Atlanta_road/'
# m.readshapefile(dir_shapefile+"Major_Roads",'road')
# plt.savefig('UFS_PM25_Atlanta.png', dpi = 600)
    
'''
3-2) NYC
'''
# m = Basemap(projection='cyl', resolution='l',llcrnrlat=40.6, urcrnrlat = 40.9,llcrnrlon=-74.1, urcrnrlon = -73.7)
# fig = plt.subplots(1,1,figsize = (18,10))
# m.pcolor(LON_ml, LAT_ml, V_month_avr,vmin =12, vmax =20,cmap = 'jet')

# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',linewidth=3,s = 300)
# cb = plt.colorbar()
# plt.clim(12,20)
# cb.set_label('PM$_{2.5}$ (\u03bcg/m$^3$)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)

# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/1_NYC_road/'
# m.readshapefile(dir_shapefile+"geo_export_24d583de-a1d3-4667-a4b0-591d6f74cf60",'nyc_road',linewidth=0.1)
# plt.savefig('UFS_PM25_NYC.png', dpi = 600)

'''
4-1-0) CONUS domain ML only
'''
m = Basemap(projection='cyl', resolution='l',llcrnrlat=25, urcrnrlat = 49,llcrnrlon=-125, urcrnrlon = -65)
fig = plt.subplots(1,1,figsize = (20,10))
m.pcolor(LON_ml, LAT_ml, V_month_final,vmin =0, vmax =20,cmap = 'jet')
m.drawcoastlines(linewidth = 1)
m.drawcountries(linewidth = 1)
m.drawstates(linewidth = 1)

cb = plt.colorbar()
plt.clim(0,20)
cb.set_label('PM$_{2.5}$ (\u03bcg/m$^3$)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)

dir_shapefile='/groups/ESS3/btang6/DATA/shapefile/USA/USA_County/'
m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')
plt.savefig('XGB_PM25_CONUS_nosea.png', dpi = 600)    
    
    
    
'''
4-2-0) California ML
'''
# m = Basemap(projection='cyl', resolution='l',llcrnrlat=31, urcrnrlat = 43,llcrnrlon=-125, urcrnrlon = -114)
# fig = plt.subplots(1,1,figsize = (15,10))
# m.pcolor(LON_ml, LAT_ml, V_month_avr,vmin =0, vmax =20,cmap = 'jet')

# m.drawcoastlines(linewidth = 1)
# m.drawcountries(linewidth = 1)
# m.drawstates(linewidth = 1)

# cb = plt.colorbar()
# plt.clim(0,20)
# cb.set_label('PM$_{2.5}$ (\u03bcg/m$^3$)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)

# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')
# plt.savefig('UFS_PM25_California.png', dpi = 600)

'''
4-2-1) California ML and OBS
'''
# m = Basemap(projection='cyl', resolution='l',llcrnrlat=31, urcrnrlat = 43,llcrnrlon=-125, urcrnrlon = -114)
# fig = plt.subplots(1,1,figsize = (15,10))
# m.pcolor(LON_ml, LAT_ml, V_month_avr,vmin =0, vmax =20,cmap = 'jet')

# m.drawcoastlines(linewidth = 1)
# m.drawcountries(linewidth = 1)
# m.drawstates(linewidth = 1)

# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='w',s = 50)
# cb = plt.colorbar()
# plt.clim(0,20)
# cb.set_label('PM$_{2.5}$ (\u03bcg/m$^3$)',fontsize = 30) 
# cb.ax.tick_params(labelsize=30)

# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')
# plt.savefig('UFS&OBS_PM25_California.png', dpi = 600) 
    
    
    
    
    
