#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:47:24 2022

@author: btang1
"""

import numpy as np
import os
from scipy import spatial
import fnmatch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.basemap import Basemap
import codecs
import datetime
from datetime import datetime as dt
# import h5py
import pandas as pd


'''
1) read-in prepared data
'''
def GetData(filename):
    file_loc = '/Volumes/Ext_Disk_1/3_NOAA_projects/NOAA_UFS_downscale/NO2_case/code/NO2_splitbysample_MUfs_ENemo_TEMPO/'
    obs_df = pd.read_excel(file_loc+filename+'.xls')
    pd.set_option('display.max_columns',None)  
        
    # TIME = list(obs_df['Time_KST'])
    LOC_NUMBER = list(obs_df['Location_number'])

    AirKorea_PM25 = list(obs_df['EPA_OBS_NO2'])
    # V1_BLH = list(obs_df['V1_BLH'])
    # V2_D2M = list(obs_df['V2_D2M'])
    # V3_E = list(obs_df['V3_E'])
    # V4_SP = list(obs_df['V4_SP'])
    # V5_T2M = list(obs_df['V5_T2M'])
    # V6_TP = list(obs_df['V6_TP'])
    # V7_U10 = list(obs_df['V7_U10'])
    # V8_V10 = list(obs_df['V8_V10'])
    # V9_AOD = list(obs_df['V9_AOD'])
    # V10_NDVI = list(obs_df['V10_NDVI'])
    # V11_LAND_USE_COVER = list(obs_df['V11_LAND_USE_COVER'])
    # V12_ELEVATION = list(obs_df['V12_ELEVATION'])
    # V13_POPULATION = list(obs_df['V13_POPULATION'])
    V14_WRF_PM25 = list(obs_df['V14_CMAQ_NO2'])
    # V15_WRF_O3 = list(obs_df['V15_WRF_O3'])
    # V16_WRF_NO = list(obs_df['V16_WRF_NO'])
    # V17_WRF_NO2 = list(obs_df['V17_WRF_NO2'])
    # V18_E_BC = list(obs_df['V18_E_BC'])
    # V19_E_SO2 = list(obs_df['V19_E_SO2'])
    # V20_E_NO = list(obs_df['V20_E_NO'])
    # V21_E_NO2 = list(obs_df['V21_E_NO2'])
    # V22_E_NH3 = list(obs_df['V22_E_NH3'])
    # V23_E_PM25 = list(obs_df['V23_E_PM25'])

    # LAT = list(obs_df['V25_Lat'])
    # LON = list(obs_df['V24_Lon'])
    # D1 = list(obs_df['V26_D1'])
    # D2 = list(obs_df['V27_D2'])
    # D3 = list(obs_df['V28_D3'])
    # D4 = list(obs_df['V29_D4'])
    # D5 = list(obs_df['V30_D5'])
    # JulianDay = list(obs_df['V31_JulianDay'])
    
    return  V14_WRF_PM25, AirKorea_PM25


V14_WRF_PM25, AirKorea_PM25 = GetData('CONUS_US_interpolated_Aug2023_NO2')


# X1 = V1_BLH
# X2 = V2_D2M
# X3 = V3_E 
# X4 = V4_SP
# X5 = V5_T2M
# X6 = V6_TP
# X7 = V7_U10
# X8 = V8_V10
# X9 = V9_AOD
# X10 = V10_NDVI
# X11 = V11_LAND_USE_COVER
# X12 = V12_ELEVATION
# X13 = V13_POPULATION
X14 = V14_WRF_PM25


# X18 = V18_E_BC
# X19 = V19_E_SO2
# X20 = V20_E_NO
# X21 = V21_E_NO2
# X22 = V22_E_NH3
# X23 = V23_E_PM25

# X24 = LAT
# X25 = LON
# X26 = D1
# X27 = D2
# X28 = D3
# X29 = D4
# X30 = D5
# X31 = JulianDay

Y =  AirKorea_PM25
# LOC_NUMBER_ = LOC_NUMBER




'''
4) final evaluation
'''
import scipy 
from sklearn.metrics import mean_squared_error   


slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(X14), np.array(Y))
rms_test = mean_squared_error(Y, X14, squared=False)
mean_bias_test = np.mean(np.array(X14)-np.array(Y))
std_obs = np.std(np.array(Y))
std_model = np.std(np.array(X14))


print('std_model', std_model)
print('std_obs', std_obs)               
print('r test',r_value_test)
# print('std test', std_err_test)
print('rms test',rms_test)
# print('mean bias test', mean_bias_test)

















