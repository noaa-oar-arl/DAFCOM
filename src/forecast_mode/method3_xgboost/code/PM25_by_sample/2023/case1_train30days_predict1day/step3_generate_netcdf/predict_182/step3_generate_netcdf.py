#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:46:06 2022

@author: btang1
"""
import numpy as np
from netCDF4 import Dataset
import joblib
import sys
import xgboost as xgb

'''
1) load saved random forest algorithm
'''
date = '182'   #182-212 is 2023/07, 213 to 243 is 2023/08
date_index = 0

dir_model = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method3_xgboost/code/PM25_by_sample/2023/case1_train30days_predict1day/step2_machine_learning/predict_'+date+'/'
loaded_xgb = joblib.load(dir_model+'xgb_model_'+date+'.joblib')

print('loaded xgb model')

dir_input_matrix = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/code/PM25_by_sample/2023/2023_07/'
sys.path.insert(1,dir_input_matrix)
'''
2) prepare input data 14
'''
from step1_Daily_Avr_InputMatrix import BLH, D2M, E, SP, T2M, TP, U10, V10, LAT, LON
from step1_Daily_Avr_InputMatrix import AOD, Land_Use_Cover, Elevation, Population
from step1_Daily_Avr_InputMatrix import UFS_AQM_PM25_daily, E_BC_daily, E_SO2_daily, E_NOX_daily, E_VOC_daily, E_NH3_daily, E_PM25_daily
from step1_Daily_Avr_InputMatrix import Lon_matrix_daily ,Lat_matrix_daily, D1_matrix_daily, D2_matrix_daily,D3_matrix_daily, D4_matrix_daily, D5_matrix_daily, day_matrix_daily

V1 = BLH
V2 = D2M
V3 = E
V4 = SP
V5 = T2M
V6 = TP
V7 = U10
V8 = V10
V9 = AOD
V11 = Land_Use_Cover
V12 = Elevation
V13 = Population
V14 = UFS_AQM_PM25_daily
V18 = E_BC_daily
V19 = E_SO2_daily
V20 = E_NOX_daily
V21 = E_VOC_daily
V22 = E_NH3_daily
V23 = E_PM25_daily
V25 = Lat_matrix_daily
V26 = Lon_matrix_daily
V27 = D1_matrix_daily
V28 = D2_matrix_daily
V29 = D3_matrix_daily
V30 = D4_matrix_daily
V31 = D5_matrix_daily
V32 = day_matrix_daily


'''
2) make daily prediction then average to monthly
'''
def GetPredDay(day_index):  #this is per day prediction
    V1_long = np.reshape(V1[day_index],(14400000,1))
    V2_long = np.reshape(V2[day_index],(14400000,1))
    V3_long = np.reshape(V3[day_index],(14400000,1))
    V4_long = np.reshape(V4[day_index],(14400000,1))
    V5_long = np.reshape(V5[day_index],(14400000,1))
    V6_long = np.reshape(V6[day_index],(14400000,1))
    V7_long = np.reshape(V7[day_index],(14400000,1))
    V8_long = np.reshape(V8[day_index],(14400000,1))
    V9_long = np.reshape(V9[day_index],(14400000,1))
    #V10_long = np.reshape(V10[day_index],(14400000,1))
    V11_long = np.reshape(V11[day_index],(14400000,1))
    V12_long = np.reshape(V12[day_index],(14400000,1))
    V13_long = np.reshape(V13[day_index],(14400000,1))
    V14_long = np.reshape(V14[day_index],(14400000,1))
    
    V18_long = np.reshape(V18[day_index],(14400000,1))
    V19_long = np.reshape(V19[day_index],(14400000,1))
    V20_long = np.reshape(V20[day_index],(14400000,1))
    V21_long = np.reshape(V21[day_index],(14400000,1))
    V22_long = np.reshape(V22[day_index],(14400000,1))
    V23_long = np.reshape(V23[day_index],(14400000,1))   
    #V24_long = np.reshape(V24[day_index],(14400000,1))
    
    V25_long = np.reshape(V25[day_index],(14400000,1))
    V26_long = np.reshape(V26[day_index],(14400000,1))
    V27_long = np.reshape(V27[day_index],(14400000,1))
    V28_long = np.reshape(V28[day_index],(14400000,1))
    V29_long = np.reshape(V29[day_index],(14400000,1))
    V30_long = np.reshape(V30[day_index],(14400000,1))
    V31_long = np.reshape(V31[day_index],(14400000,1))
    V32_long = np.reshape(V32[day_index],(14400000,1))
    
    input_test = np.reshape(np.transpose([V1_long,V2_long,V3_long,V4_long,V5_long,V6_long,V7_long,V8_long,V9_long,V11_long,V12_long,V13_long,V14_long,V18_long,V19_long,V20_long,V21_long,V22_long,V23_long,V25_long,V26_long,V27_long,V28_long,V29_long,V30_long,V31_long,V32_long]),(14400000,27))
    
    
    Y_hat_long = loaded_xgb.predict(input_test)
    Y_hat = np.reshape(Y_hat_long,(2400,6000)) 

    return Y_hat          


'''
supplemental save in netcdf file
'''
Y_hat_month_daily = []
for i in range(1):
    #print(i)
    Y_hat_day = GetPredDay(date_index)
    Y_hat_month_daily.append(Y_hat_day)
    
lat_WestUS = [49- 0.01*X for X in range(2400)]
lon_WestUS = [-125+ 0.01*X for X in range(6000)]    
    
ft = Dataset(('forecast_'+date+'_pm25_0p01'+'.nc'),'w',format = 'NETCDF4')   
nlat = ft.createDimension('latitude',2400)
nlon = ft.createDimension('longitude',6000)
nlayer = ft.createDimension('time',1)


lat_new_nc = ft.createVariable('latitude','f8',('latitude'))
lat_new_nc.units =''
lat_new_nc.long_name ='latitude'

lon_new_nc = ft.createVariable('longitude','f8',('longitude'))
lon_new_nc.units =''
lon_new_nc.long_name ='longitude'

variable_new = ft.createVariable('pm25','f8',('time','latitude','longitude'))  
variable_new .units ='ug/m^3'                                               
variable_new .description =''

lat_new_nc[:] = np.array(lat_WestUS )
lon_new_nc[:] = np.array(lon_WestUS)
variable_new[:,:,:] = np.array(Y_hat_month_daily )

ft.close()


            

























