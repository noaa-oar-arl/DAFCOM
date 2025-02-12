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
from step1_Daily_Avr_InputMatrix import LAT, LON,UFS_AQM_PM25_daily,Forecast_PM25_daily
from step2_Daily_Avr_GS_PM25 import LOC_NUMBER_OBS_FINAL, PM_OBS_FINAL, TIME_OBS_FINAL,LAT_OBS_FINAL,LON_OBS_FINAL

number_of_days = 31
month_of_year ='08'


'''
2) define model
'''

def GetCams(time_input,lon_input,lat_input, variable_input):
    
    lat_obs = lat_input
    lon_obs = lon_input
    
    lat_cams  = LAT #2400
    lon_cams  = LON #6000
    variable_cams = variable_input  # (31,2400,6000)                                       
    
    c = 'no_value'
    for i in range(1,2400):         
        if (lat_cams[i-1]-lat_obs)*(lat_cams[i]-lat_obs) < 0: 
            
            for j in range(1,6000):          
                if (lon_cams[j-1]-lon_obs)*(lon_cams[j]-lon_obs) < 0:     
                    
                    x1 = np.abs((lon_cams[j-1]-lon_obs)/(lon_cams[j-1]-lon_cams[j]))
                    x2 = np.abs((lon_cams[j]-lon_obs)/(lon_cams[j-1]-lon_cams[j]))
                    
                    y1 = np.abs((lat_cams[i-1]-lat_obs)/(lat_cams[i-1]-lat_cams[i]))
                    y2 = np.abs((lat_cams[i]-lat_obs)/(lat_cams[i-1]-lat_cams[i]))
                    
                    for k in range(number_of_days):             
                        if time_input.day == k+1:
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
V1 = UFS_AQM_PM25_daily
V2 = Forecast_PM25_daily

V1_final_list = []   
V2_final_list = []   

for i in range(len(TIME_OBS_FINAL)):             # total points, here can specify [start:end_i]
    loc_number_here = LOC_NUMBER_OBS_FINAL[i]
    lat_here = LAT_OBS_FINAL[i]
    lon_here = LON_OBS_FINAL[i]
   
    V1_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V1)
    V2_final =  GetCams(TIME_OBS_FINAL[i],lon_here,lat_here, V2)    
    
    print(i,TIME_OBS_FINAL[i],lon_here,lat_here,V1_final)
    
    V1_final_list.append(V1_final)
    V2_final_list.append(V2_final)

'''
3) write output into excel
'''
wb = Workbook()
sheet1 = wb.add_sheet('EPA_PM25')

sheet1.write(0,0,'no')
sheet1.write(0,1,'Time_UTC')
sheet1.write(0,2,'Location_number')
sheet1.write(0,3,'EPA_OBS_PM25')

sheet1.write(0,5,'V1_UFS_AQM_PM25')
sheet1.write(0,6,'V2_Forecast_PM25')

for i in range(len(V1_final_list)):  
    sheet1.write(i+1,1,TIME_OBS_FINAL[i])
    sheet1.write(i+1,2,LOC_NUMBER_OBS_FINAL[i])
    sheet1.write(i+1,3,PM_OBS_FINAL[i])
    
    sheet1.write(i+1,5,V1_final_list[i])
    sheet1.write(i+1,6,V2_final_list[i])
    
wb.save('Evaluate_PM25_2023_'+month_of_year+'_forecast_vs_OBS.xls')
    
            
            
        





























