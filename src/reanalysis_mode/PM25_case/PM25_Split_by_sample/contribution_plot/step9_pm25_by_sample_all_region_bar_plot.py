#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:28:17 2019

@author: tangbeiming
"""

import numpy as np
import pandas as pd
# import xarray as xr
import matplotlib.pyplot as plt
# import seaborn as sns


'''
1) import data
'''
Input_list = ['BLH','D2m', 'Evaporation', 'SP', 'T2m',
              'Precipitation', 'Wind-u','Wind-v', 'AOD', 'NDVI', 
              'LUC','Elevation', 'Population', 'UFS-AQM', 'Emis_BC', 
              'Emis_SO2','Emis_VOC','Emis_NOx', 'Emis_NH3', 'Emis_PM25', 
              'Spatial_info','Julian_Day']



pm25_sample_2016WM_base=[3,4,1,2,3,
                         3,3,3,19,1,
                         0,1,0,25,1,
                         1,1,0,1,0,
                         16,9]
# pm25_sample_2016WM_noCTM=[3,3,10,2,6,
#                           4,6,5,9,1,
#                           0,1,1,0,1,
#                           1,1,1,1,1,
#                           9,36]
# pm25_sample_2016WM_noJulian=[3,4,14,2,9,
#                              5,7,6,11,1,
#                              1,1,1,21,1,
#                              1,1,1,1,1,
#                              10,0]
# pm25_sample_2016WM_noSpatial=[3,3,8,2,6,
#                               4,5,4,7,1,
#                               1,1,1,15,1,
#                               1,1,1,1,1,
#                               0,32]




'''
0) initialization
'''
v1 = pm25_sample_2016WM_base
# v2 = pm25_sample_2016WM_noCTM
# v3 = pm25_sample_2016WM_noJulian
# v4 = pm25_sample_2016WM_noSpatial

save_name = 'pm25_by_sample'


'''
2)bar plot
'''
fig,ax = plt.subplots(figsize=(40,20))
plt.rc('ytick', labelsize=40)    #change y label fontsize
plt.rc('xtick', labelsize=30) 

positions1 = [i for i in range(len(v1))]
# positions2 = [i-0.4 for i in range(len(v2))]
# positions3 = [i-0.2 for i in range(len(v3))]
# positions4 = [i for i in range(len(v4))]

labels1 = np.array(Input_list)

plt.bar(positions1 ,v1,width =0.5,label='2023 June PM$_{2.5}$',color='#ff7f00')
# plt.bar(positions2 ,v2,width =0.2,label='2016WM noCTM',color='#377eb8')
# plt.bar(positions3 ,v3,width =0.2,label='2016WM noJulian',color='#4daf4a')
# plt.bar(positions4 ,v4,width =0.2,label='2016WM noSpatial',color='#999999')

ax.set_xticks(positions1, labels=labels1)
plt.ylim(0,30)
plt.ylabel('Contribution Percentage(%)',fontsize=40)
plt.xticks(rotation = 60)
plt.legend(loc='upper left',fontsize=40)
plt.savefig(save_name+'_all_input_bar_plot.png', dpi=600)










