#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:49:54 2022

@author: btang1
"""


import numpy as np
from xlwt import Workbook

'''
1) import data
'''
from step1_Daily_Avr_InputMatrix import BLH, D2M, E, SP, T2M, TP, U10, V10, LAT, LON
from step1_Daily_Avr_InputMatrix import AOD, Land_Use_Cover, Elevation, Population
from step1_Daily_Avr_InputMatrix import UFS_AQM_PM25_daily, E_BC_daily, E_SO2_daily, E_NOX_daily, E_VOC_daily, E_NH3_daily, E_PM25_daily
from step1_Daily_Avr_InputMatrix import Lon_matrix_daily ,Lat_matrix_daily, D1_matrix_daily, D2_matrix_daily,D3_matrix_daily, D4_matrix_daily, D5_matrix_daily, day_matrix_daily
from step2_Daily_Avr_GS_PM25 import LOC_NUMBER_OBS_FINAL, PM_OBS_FINAL, TIME_OBS_FINAL,LAT_OBS_FINAL,LON_OBS_FINAL

number_of_days = 30
month_of_year ='06'


'''
2) define model
'''

def GetCams(time_input,lon_input,lat_input, variable_input):
    
    lat_obs = lat_input
    lon_obs = lon_input
    
    lat_cams  = LAT #2500
    lon_cams  = LON #3500
    variable_cams = variable_input  # (30,2500,3500)                                       
    
    c = 'no_value'
    for i in range(1,2400):            #600
        if (lat_cams[i-1]-lat_obs)*(lat_cams[i]-lat_obs) < 0: 
            
            for j in range(1,6000):   #400         
                if (lon_cams[j-1]-lon_obs)*(lon_cams[j]-lon_obs) < 0:     
                    
                    x1 = np.abs((lon_cams[j-1]-lon_obs)/(lon_cams[j-1]-lon_cams[j]))
                    x2 = np.abs((lon_cams[j]-lon_obs)/(lon_cams[j-1]-lon_cams[j]))
                    
                    y1 = np.abs((lat_cams[i-1]-lat_obs)/(lat_cams[i-1]-lat_cams[i]))
                    y2 = np.abs((lat_cams[i]-lat_obs)/(lat_cams[i-1]-lat_cams[i]))
                    
                    for k in range(number_of_days):                                        #CHANGE HERE !!!!!!!!!!!!
                        if time_input.day == k+1:
                            # print('yes')
                            a = x2 * variable_cams[k][i-1][j-1] + x1 * variable_cams[k][i-1][j]
                            b = x2 * variable_cams[k][i][j-1] + x1 * variable_cams[k][i][j]
                            c = y1 * b + y2 * a  
                            break
                    break
            break
            
    return c
'''
3) 2d-interpolation MAIN
'''
V1 = BLH
V2 = D2M
V3 = E
V4 = SP
V5 = T2M
V6 = TP
V7 = U10
V8 = V10
V9 = AOD
#V10 = NDVI
V11 = Land_Use_Cover
V12 = Elevation
V13 = Population
V14 = UFS_AQM_PM25_daily
# V15 = CAMS_o3_daily
# V16 = CAMS_no_daily
# V17 = CAMS_no2_daily
V18 = E_BC_daily
V19 = E_SO2_daily
V20 = E_NOX_daily
V21 = E_VOC_daily
V22 = E_NH3_daily
V23 = E_PM25_daily
#V24 = TROP_NO2
V25 = Lon_matrix_daily
V26 = Lat_matrix_daily
V27 = D1_matrix_daily
V28 = D2_matrix_daily
V29 = D3_matrix_daily
V30 = D4_matrix_daily
V31 = D5_matrix_daily
V32 = day_matrix_daily








V1_final_list = []   
V2_final_list = []   
V3_final_list = []   
V4_final_list = []   
V5_final_list = []   
V6_final_list = []   
V7_final_list = []   
V8_final_list = []   
V9_final_list = []   
#V10_final_list = []  
V11_final_list = []   
V12_final_list = []   
V13_final_list = []   
V14_final_list = []   
# V15_final_list = []  
# V16_final_list = []  
# V17_final_list = []  
V18_final_list = []  
V19_final_list = []  
V20_final_list = []  
V21_final_list = []  
V22_final_list = []  
V23_final_list = []  
#V24_final_list = []  
V25_final_list = []  
V26_final_list = []  
V27_final_list = []  
V28_final_list = []  
V29_final_list = []  
V30_final_list = []  
V31_final_list = []  
V32_final_list = []  

for i in range(len(TIME_OBS_FINAL)):             # 10621 points, here can specify [start:end_i]
# for i in range(100):    
    loc_number_here = LOC_NUMBER_OBS_FINAL[i]
    lat_here = LAT_OBS_FINAL[i]
    lon_here = LON_OBS_FINAL[i]
    # print(TIME_OBS_FINAL[i],loc_number_here)
   
    V1_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V1)
    V2_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V2)    
    V3_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V3)    
    V4_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V4)    
    V5_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V5)    
    V6_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V6)
    V7_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V7)    
    V8_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V8)    
    V9_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V9)    
    #V10_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V10)   
    V11_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V11)
    V12_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V12)    
    V13_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V13)    
    V14_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V14) 
    # V15_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V15)
    # V16_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V16)
    # V17_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V17)
    V18_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V18)
    V19_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V19)
    V20_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V20)
    V21_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V21)
    V22_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V22)
    V23_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V23)
    #V24_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V24)
    V25_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V25)
    V26_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V26)
    V27_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V27)
    V28_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V28)
    V29_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V29)
    V30_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V30)
    V31_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V31)
    V32_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V32)
    
    print(i,TIME_OBS_FINAL[i],lon_here,lat_here,V1_final)
    
    V1_final_list.append(V1_final)
    V2_final_list.append(V2_final)
    V3_final_list.append(V3_final)
    V4_final_list.append(V4_final)
    V5_final_list.append(V5_final)    
    V6_final_list.append(V6_final)
    V7_final_list.append(V7_final)
    V8_final_list.append(V8_final)
    V9_final_list.append(V9_final)
    #V10_final_list.append(V10_final)   
    V11_final_list.append(V11_final)
    V12_final_list.append(V12_final)
    V13_final_list.append(V13_final)
    V14_final_list.append(V14_final)  
    # V15_final_list.append(V15_final)
    # V16_final_list.append(V16_final)
    # V17_final_list.append(V17_final)
    V18_final_list.append(V18_final)
    V19_final_list.append(V19_final)
    V20_final_list.append(V20_final)
    V21_final_list.append(V21_final)
    V22_final_list.append(V22_final)
    V23_final_list.append(V23_final)
    #V24_final_list.append(V24_final)
    V25_final_list.append(V25_final)
    V26_final_list.append(V26_final)
    V27_final_list.append(V27_final)
    V28_final_list.append(V28_final)
    V29_final_list.append(V29_final)
    V30_final_list.append(V30_final)
    V31_final_list.append(V31_final)
    V32_final_list.append(V32_final)
    
            
'''
3) write output into excel
'''
#from xlwt import Workbook


wb = Workbook()
sheet1 = wb.add_sheet('EPA_no2')

sheet1.write(0,0,'no')
sheet1.write(0,1,'Time_UTC')
sheet1.write(0,2,'Location_number')
sheet1.write(0,3,'EPA_OBS_NO2')

sheet1.write(0,5,'V1_BLH')
sheet1.write(0,6,'V2_D2M')
sheet1.write(0,7,'V3_E')
sheet1.write(0,8,'V4_SP')
sheet1.write(0,9,'V5_T2M') 
sheet1.write(0,10,'V6_TP')
sheet1.write(0,11,'V7_U10')
sheet1.write(0,12,'V8_V10')
sheet1.write(0,13,'V9_AOD')
#sheet1.write(0,14,'V10_NDVI') 
sheet1.write(0,15,'V11_LAND_USE_COVER')
sheet1.write(0,16,'V12_ELEVATION')
sheet1.write(0,17,'V13_POPULATION')
sheet1.write(0,18,'V14_UFS_AQM_PM25')
# sheet1.write(0,19,'V15_CAMS_O3')
# sheet1.write(0,20,'V16_CAMS_NO')
# sheet1.write(0,21,'V17_CAMS_NO2')
sheet1.write(0,22,'V18_E_BC')
sheet1.write(0,23,'V19_E_SO2')
sheet1.write(0,24,'V20_E_NOX')
sheet1.write(0,25,'V21_E_VOC')
sheet1.write(0,26,'V22_E_NH3')
sheet1.write(0,27,'V23_E_PM25')
#sheet1.write(0,28,'V24_TROP_NO2')
sheet1.write(0,29,'LON')
sheet1.write(0,30,'LAT')
sheet1.write(0,31,'D1')
sheet1.write(0,32,'D2')
sheet1.write(0,33,'D3')
sheet1.write(0,34,'D4')
sheet1.write(0,35,'D5')
sheet1.write(0,36,'Julian_day')


for i in range(len(V1_final_list)):  
    # sheet1.write(i+1,0,i+1)
    sheet1.write(i+1,1,TIME_OBS_FINAL[i])
    sheet1.write(i+1,2,LOC_NUMBER_OBS_FINAL[i])
    sheet1.write(i+1,3,PM_OBS_FINAL[i])
    
    sheet1.write(i+1,5,V1_final_list[i])
    sheet1.write(i+1,6,V2_final_list[i])
    sheet1.write(i+1,7,V3_final_list[i])
    sheet1.write(i+1,8,V4_final_list[i])
    sheet1.write(i+1,9,V5_final_list[i])
    sheet1.write(i+1,10,V6_final_list[i])
    sheet1.write(i+1,11,V7_final_list[i])
    sheet1.write(i+1,12,V8_final_list[i])
    sheet1.write(i+1,13,V9_final_list[i])
    #sheet1.write(i+1,14,V10_final_list[i])
    sheet1.write(i+1,15,V11_final_list[i])
    sheet1.write(i+1,16,V12_final_list[i])
    sheet1.write(i+1,17,V13_final_list[i])
    sheet1.write(i+1,18,V14_final_list[i])
    # sheet1.write(i+1,19,V15_final_list[i])
    # sheet1.write(i+1,20,V16_final_list[i])
    # sheet1.write(i+1,21,V17_final_list[i])
    sheet1.write(i+1,22,V18_final_list[i])
    sheet1.write(i+1,23,V19_final_list[i])
    sheet1.write(i+1,24,V20_final_list[i])
    sheet1.write(i+1,25,V21_final_list[i])
    sheet1.write(i+1,26,V22_final_list[i])
    sheet1.write(i+1,27,V23_final_list[i])
    #sheet1.write(i+1,28,V24_final_list[i])
    sheet1.write(i+1,29,V25_final_list[i])
    sheet1.write(i+1,30,V26_final_list[i])
    sheet1.write(i+1,31,V27_final_list[i])
    sheet1.write(i+1,32,V28_final_list[i])
    sheet1.write(i+1,33,V29_final_list[i])
    sheet1.write(i+1,34,V30_final_list[i])   
    sheet1.write(i+1,35,V31_final_list[i])   
    sheet1.write(i+1,36,V32_final_list[i]) 

    
wb.save('CONUS_US_interpolated_2023_'+month_of_year+'_PM25.xls')
    
            
            
        





























