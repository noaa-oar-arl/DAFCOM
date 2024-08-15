#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:07:08 2022

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

'''
1) PM UFS
'''
print('pm get')
dir_ = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/data/6_UFS_AQM/'

pattern_0 = 'Surface_Map_pm25.nc'
filename_0 = dir_ + pattern_0
f0 = Dataset(filename_0,'r')
LAT = f0.variables['lat'][:] # (600)
LON = f0.variables['lon'][:] #(400)
PM25 = f0.variables['pm25'][:]

v_input = PM25 #CHANGE
V_month_avr = v_input[0]
for i in range(1,len(v_input)):
    V_month_avr += v_input[i]
    
V_month_avr = V_month_avr /len(PM25)


'''
1) PLOT
'''
'''
4-1) CONUS domain
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='l',llcrnrlat=25, urcrnrlat = 49,llcrnrlon=-125, urcrnrlon = -65) 
fig = plt.subplots(1,1,figsize = (20,10)) 
m.pcolor(LON, LAT, V_month_avr,vmin = 0, vmax = 35,cmap = 'jet') 

m.drawcoastlines(linewidth = 2)
m.drawcountries(linewidth = 2)
m.drawstates(linewidth = 2)

                                  
#plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='w',linewidth=3,s = 50)
cb = plt.colorbar()
plt.clim(0,35)
cb.set_label('PM$_{2.5}$ (\u03bcg/$m^3$)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  


#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')       
plt.savefig('UFS_pm25__CONUS.png', dpi = 600)
    
'''
4-3) LA
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='l',llcrnrlat=33.7, urcrnrlat = 34.2,llcrnrlon=-118.6, urcrnrlon = -117.8) 
fig = plt.subplots(1,1,figsize = (15,10)) 
m.pcolor(LON, LAT, V_month_avr,vmin = 0, vmax = 20,cmap = 'jet') 

                                  
#plot airkorea obs scatter maps
# m.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',linewidth=3,s = 300)
# cb = plt.colorbar(location= 'bottom')
cb = plt.colorbar()
plt.clim(0,20)
cb.set_label('$PM_{2.5}$ (\u03bcg/$m^3$)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  


#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/10_LA_road/'
m.readshapefile(dir_shapefile+"tl_2018_06037_roads",'la_roads',linewidth = 0.3)
plt.savefig('UFS_pm25_LA.png', dpi = 600)  
    
    
'''
4-4) denver
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='l',llcrnrlat=39.65, urcrnrlat = 39.80,llcrnrlon=-105.05, urcrnrlon = -104.9) 
fig = plt.subplots(1,1,figsize = (15,10)) 
m.pcolor(LON, LAT, V_month_avr,vmin = 0, vmax = 15,cmap = 'jet') 
                                
#plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',linewidth=3,s = 300)
cb = plt.colorbar()
plt.clim(0,15)
cb.set_label('PM$_{2.5}$ (\u03bcg/$m^3$)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  

#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')    
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/7_Denver_road/'
m.readshapefile(dir_shapefile+"tl_2019_08031_roads",'denver_roads',linewidth = 0.3)  
plt.savefig('UFS_pm25__Denver.png', dpi = 600)    
    
'''
4-2) DC
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='h',llcrnrlat=38.8, urcrnrlat = 39.0,llcrnrlon=-77.13, urcrnrlon = -76.9) 
fig = plt.subplots(1,1,figsize = (18,10)) 
m.pcolor(LON, LAT, V_month_avr,vmin =5 , vmax = 20,cmap = 'jet') 
                                
#plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',
            # edgecolors='k',linewidth=3,s = 300)
cb = plt.colorbar()
plt.clim(5,20)
cb.set_label('PM$_{2.5}$ (\u03bcg/$m^3$)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  

m.drawcountries(linewidth = 1)
# m.drawstates(linewidth = 1)
#add shape file
# dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/USA_County/'
# m.readshapefile(dir_shapefile+"tl_rd22_us_county",'US_states')    
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/2_DC_road/'
m.readshapefile(dir_shapefile+"Roads",'DC_road',linewidth=0.1)
plt.savefig('UFS_pm25_DC.png', dpi = 600)   
    


'''
4-4) Alberquque, NM
'''
# #plot monthly averge map 
m = Basemap(projection='cyl', resolution='i',llcrnrlat=35.00, urcrnrlat = 35.22,llcrnrlon=-106.78, urcrnrlon = -106.48) # Alberqueuq, NM
fig = plt.subplots(1,1,figsize = (15,10)) 
m.pcolor(LON, LAT,V_month_avr,vmin = 0, vmax = 10,cmap = 'jet') 

# m.drawcoastlines(linewidth = 2)
m.drawcountries(linewidth = 2)
m.drawstates(linewidth = 2)

                                  
#plot airkorea obs scatter maps
# plt.scatter(np.array(Lon_bystation),np.array(Lat_bystation), c = np.array(Y_diff_bystation),cmap = 'jet',edgecolors='k',linewidth=3,s = 300)
cb = plt.colorbar()
plt.clim(0,10)
cb.set_label('$PM_{2.5}$ (\u03bcg/$m^3$)',fontsize = 30) 
cb.ax.tick_params(labelsize=30)  


#add shape file
dir_shapefile='/Volumes/Ext_Disk_2/Data/shapefiles/USA/Alberquque_NM/'
m.readshapefile(dir_shapefile+"alberquque_streets",'alberquque_streets')      
plt.savefig('UFS_pm25_Alberquque_NM.png', dpi = 600)



    
    
    
    
    
    
    
    
    
    
    