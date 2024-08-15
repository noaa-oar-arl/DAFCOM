#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:09:53 2022

@author: btang1
"""


import numpy as np
import os
from scipy import spatial
import fnmatch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
# import codecs
import datetime
from datetime import datetime as dt
import h5py
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import joblib
# import scipy 


'''
1) import re-grid data
'''
from All_wrfinput_variables import WRF_pm25_daily,WRF_o3_daily,WRF_no2_daily,WRF_bc_daily, LAT, LON

v_input = WRF_pm25_daily #CHANGE
V_month_avr = v_input[0]
for i in range(1,len(v_input)):
    if i != 9:
        V_month_avr += v_input[i]
    
V_month_avr = V_month_avr /30


'''
2) plot  re-grid data
'''


m = Basemap(projection='cyl', resolution='l',llcrnrlat=33, urcrnrlat = 39,llcrnrlon=126, urcrnrlon = 130) 
fig = plt.subplots(1,1,figsize = (12,12)) 
# m.pcolor(LON, LAT, V_month_avr, cmap = 'jet') 
m.pcolor(LON, LAT, V_month_avr, vmin = 0, vmax = 50, cmap = 'jet') 
cb = m.colorbar()
cb.set_label('WRF_pm25 (ug/$m^3$)')  #CHANGE                                     

dir_shapefile='/Users/btang1/Desktop/obj3_ML/data_wrf/South_Korea_shapefile/stanford_1level_korea/'
m.readshapefile(dir_shapefile+"dk009rq9138",'states',default_encoding='iso-8859-15')
m.drawparallels(np.arange(33, 39, 2), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(126, 130, 2), labels=[0, 0, 0, 1])

plt.title('0.01 degreee lat *0.01 degree lon \nre-grid WRF_pm25\n May-2016') #CHANGE
plt.savefig('re-grid_6_WRF_pm25.png', dpi = 300)#CHANGE


