#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:47:24 2022

@author: btang1
"""

import numpy as np
from netCDF4 import Dataset
import pandas as pd


'''
1) read-in prepared data
'''
def GetData(filename):
    file_loc = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method1_entire_month_train/code/PM25_by_sample/2023/2023_08/step8_evaluate_forecast_result/forecast202308_train202306_case/'  #change here
    obs_df = pd.read_excel(file_loc+filename+'.xls')
    pd.set_option('display.max_columns',None)  
        
    TIME = list(obs_df['Time_UTC'])
    LOC_NUMBER = list(obs_df['Location_number'])

    EPA_OBS_PM25 = list(obs_df['EPA_OBS_PM25'])
    V1_UFS_AQM_PM25 = list(obs_df['V1_UFS_AQM_PM25'])
    V2_Forecast_PM25_train202307_predict202308 = list(obs_df['V2_Forecast_PM25_train202307_predict202308'])  #change here
    
    return LOC_NUMBER,EPA_OBS_PM25,V1_UFS_AQM_PM25,V2_Forecast_PM25_train202307_predict202308  #change here


LOC_NUMBER,EPA_OBS_PM25,V1_UFS_AQM_PM25,V2_Forecast_PM25_train202307_predict202308 = GetData('Evaluate_PM25_2023_08_forecast_vs_OBS') #change here

'''
4) final evaluation
'''
import scipy 
from sklearn.metrics import mean_squared_error   

X1 = EPA_OBS_PM25
X2 = V2_Forecast_PM25_train202307_predict202308  #Change here
X3 = V1_UFS_AQM_PM25

slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(X1), np.array(X2))
rms_test = mean_squared_error(X1, X2, squared=False)
mean_bias= np.mean(np.array(X1)-np.array(X2))
std_model = np.std(X2)
std_obs = np.std(X1)
print('this is OBS vs. Forecast')
print('r test',r_value_test)
print('rms test',rms_test)
print('mean bias test', mean_bias)
print('std model =',std_model)
print('std obs =',std_obs)

print('/n')
print('/n')

slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(X1), np.array(X3))
rms_test = mean_squared_error(X1, X3, squared=False)
mean_bias= np.mean(np.array(X1)-np.array(X3))
std_model = np.std(X3)
std_obs = np.std(X1)
print('this is OBS vs. Original UFS')
print('r test',r_value_test)
print('rms test',rms_test)
print('mean bias test', mean_bias)
print('std model =',std_model)
print('std obs =',std_obs)
























