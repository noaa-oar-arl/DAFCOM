#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:49:54 2022

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
import h5py
import pandas as pd
import statistics as stat


'''
1) model 
'''
def Modify():
    file_loc = './'
    obs_df = pd.read_excel(file_loc+'23DATA_AIRKOREA_INTERPOLATED_2021May.xls')
    pd.set_option('display.max_columns',None)  
    
    time_kst = obs_df['Time_KST']
    loc_num = obs_df['Location_number']
    value_obs = obs_df['AirKorea_OBS_PM25']
    v1_blh = obs_df['V1_BLH']
    v2_d2m = obs_df['V2_D2M']
    v3_e = obs_df['V3_E']
    v4_sp = obs_df['V4_SP']
    v5_t2m = obs_df['V5_T2M']
    v6_tp = obs_df['V6_TP']
    v7_u10 = obs_df['V7_U10']
    v8_v10 = obs_df['V8_V10']
    v9_aod = obs_df['V9_AOD']
    v10_ndvi = obs_df['V10_NDVI']
    v11_land_use_cover = obs_df['V11_LAND_USE_COVER']
    v12_elevation = obs_df['V12_ELEVATION']
    v13_population = obs_df['V13_POPULATION']
    v14_wrf_pm25 = obs_df['V14_CAMS_PM25']
    v15_wrf_o3 = obs_df['V15_CAMS_O3']
    v16_wrf_no = obs_df['V16_CAMS_NO']
    v17_wrf_no2 = obs_df['V17_CAMS_NO2']
    v18_e_bc = obs_df['V18_E_BC']
    v19_e_so2 = obs_df['V19_E_SO2']
    v20_e_no = obs_df['V20_E_NOX']
    v21_e_no2 = obs_df['V21_E_NMVOC']
    v22_e_nh3 = obs_df['V22_E_NH3']
    v23_e_pm25 = obs_df['V23_E_PM25']
    v24_lon = obs_df['LON']
    v25_lat = obs_df['LAT']
    v26_d1  = obs_df['D1']
    v27_d2  = obs_df['D2']
    v28_d3  = obs_df['D3']
    v29_d4  = obs_df['D4']
    v30_d5  = obs_df['D5']
    v31_julian_day = obs_df['Julian_day']
    
    Time_KST = []
    LOC_NUM = []
    Value_OBS = []
    V1 = []
    V2 = []
    V3 = []
    V4 = []
    V5 = []
    V6 = []
    V7 = []
    V8 = []
    V9 = []
    V10 = []
    V11 = []
    V12 = []
    V13 = []
    V14 = [] 
    V15 = []
    V16 = []
    V17 = []
    V18 = []
    V19 = []
    V20 = []
    V21 = []
    V22 = []
    V23 = []
    V24 = []
    V25 = []
    V26 = []
    V27 = []
    V28 = []
    V29 = []
    V30 = []
    V31 = []
    
    for i in range(len(v31_julian_day)):
        if v31_julian_day[i] != 140:
            if v31_julian_day[i] != 'no_value':
                
                Time_KST.append(time_kst[i])
                LOC_NUM.append(loc_num[i])
                Value_OBS.append(value_obs[i])
                V1.append(v1_blh[i])
                V2.append(v2_d2m[i])
                V3.append(v3_e[i])
                V4.append(v4_sp[i])
                V5.append(v5_t2m[i])
                V6.append(v6_tp[i])
                V7.append(v7_u10[i])
                V8.append(v8_v10[i])
                V9.append(v9_aod[i])
                V10.append(v10_ndvi[i])
                V11.append(v11_land_use_cover[i])
                V12.append(v12_elevation[i])
                V13.append(v13_population[i])
                V14.append(v14_wrf_pm25[i])
                V15.append(v15_wrf_o3[i])
                V16.append(v16_wrf_no[i])
                V17.append(v17_wrf_no2[i])
                V18.append(v18_e_bc[i])
                V19.append(v19_e_so2[i])
                V20.append(v20_e_no[i])
                V21.append(v21_e_no2[i])
                V22.append(v22_e_nh3[i])
                V23.append(v23_e_pm25[i])
                V24.append(v24_lon[i])
                V25.append(v25_lat[i])
                V26.append(v26_d1[i])
                V27.append(v27_d2[i])
                V28.append(v28_d3[i])
                V29.append(v29_d4[i])
                V30.append(v30_d5[i])
                V31.append(v31_julian_day[i])
                
                    
                    
    return Time_KST, LOC_NUM, Value_OBS, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,V15, V16, V17, V18, V19, V20, V21, V22, V23, V24,V25,V26, V27, V28, V29, V30, V31 


Time_KST_new, LOC_NUM_new, Value_OBS_new, V1_new, V2_new, V3_new, V4_new, V5_new, V6_new, V7_new, V8_new, V9_new, V10_new, V11_new, V12_new, V13_new, V14_new,V15_new,V16_new,V17_new,V18_new,V19_new,V20_new,V21_new,V22_new,V23_new,V24_new,V25_new,V26_new,V27_new,V28_new,V29_new,V30_new,V31_new= Modify()



            
'''
3) write output into excel
'''
from xlwt import Workbook


wb = Workbook()
sheet1 = wb.add_sheet('23data_airkorea')

sheet1.write(0,0,'no')
sheet1.write(0,1,'Time_KST')
sheet1.write(0,2,'Location_number')
sheet1.write(0,3,'AirKorea_OBS_PM25')

sheet1.write(0,5,'V1_BLH')
sheet1.write(0,6,'V2_D2M')
sheet1.write(0,7,'V3_E')
sheet1.write(0,8,'V4_SP')
sheet1.write(0,9,'V5_T2M') 
sheet1.write(0,10,'V6_TP')
sheet1.write(0,11,'V7_U10')
sheet1.write(0,12,'V8_V10')
sheet1.write(0,13,'V9_AOD')
sheet1.write(0,14,'V10_NDVI') 
sheet1.write(0,15,'V11_LAND_USE_COVER')
sheet1.write(0,16,'V12_ELEVATION')
sheet1.write(0,17,'V13_POPULATION')
sheet1.write(0,18,'V14_CAMS_PM25')
sheet1.write(0,19,'V15_CAMS_O3')
sheet1.write(0,20,'V16_CAMS_NO')
sheet1.write(0,21,'V17_CAMS_NO2')
sheet1.write(0,22,'V18_E_BC')
sheet1.write(0,23,'V19_E_SO2')
sheet1.write(0,24,'V20_E_NOX')
sheet1.write(0,25,'V21_E_NMVOC')
sheet1.write(0,26,'V22_E_NH3')
sheet1.write(0,27,'V23_E_PM25')
sheet1.write(0,28,'LON')
sheet1.write(0,29,'LAT')
sheet1.write(0,30,'D1')
sheet1.write(0,31,'D2')
sheet1.write(0,32,'D3')
sheet1.write(0,33,'D4')
sheet1.write(0,34,'D5')
sheet1.write(0,35,'Julian_day')



for i in range(len(V1_new)):  
    # sheet1.write(i+1,0,i+1)
    sheet1.write(i+1,1,Time_KST_new[i])
    sheet1.write(i+1,2,float(LOC_NUM_new[i]))
    sheet1.write(i+1,3,Value_OBS_new[i])
    
    sheet1.write(i+1,5,V1_new[i])
    sheet1.write(i+1,6,V2_new[i])
    sheet1.write(i+1,7,V3_new[i])
    sheet1.write(i+1,8,V4_new[i])
    sheet1.write(i+1,9,V5_new[i])
    sheet1.write(i+1,10,V6_new[i])
    sheet1.write(i+1,11,V7_new[i])
    sheet1.write(i+1,12,V8_new[i])
    sheet1.write(i+1,13,V9_new[i])
    sheet1.write(i+1,14,V10_new[i])
    sheet1.write(i+1,15,V11_new[i])
    sheet1.write(i+1,16,V12_new[i])
    sheet1.write(i+1,17,V13_new[i])
    sheet1.write(i+1,18,V14_new[i])
    sheet1.write(i+1,19,V15_new[i])
    sheet1.write(i+1,20,V16_new[i])
    sheet1.write(i+1,21,V17_new[i])
    sheet1.write(i+1,22,V18_new[i])
    sheet1.write(i+1,23,V19_new[i])
    sheet1.write(i+1,24,V20_new[i])
    sheet1.write(i+1,25,V21_new[i])
    sheet1.write(i+1,26,V22_new[i])
    sheet1.write(i+1,27,V23_new[i])
    
    sheet1.write(i+1,28,V24_new[i])
    sheet1.write(i+1,29,V25_new[i])
    sheet1.write(i+1,30,V26_new[i])
    sheet1.write(i+1,31,V27_new[i])
    sheet1.write(i+1,32,V28_new[i])
    sheet1.write(i+1,33,V29_new[i])
    sheet1.write(i+1,34,V30_new[i])
    sheet1.write(i+1,35,V31_new[i])    

    
wb.save('GEMS_PM25_31DATA_AIRKOREA_INTERPOLATED_2021May_modified.xls')
    
    
            
            
        





























