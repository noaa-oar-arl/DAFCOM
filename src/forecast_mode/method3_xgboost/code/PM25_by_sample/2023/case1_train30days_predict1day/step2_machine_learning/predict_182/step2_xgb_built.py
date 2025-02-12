#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:47:24 2022

@author: btang1
"""

import numpy as np
from netCDF4 import Dataset
import pandas as pd
import xgboost as xgb
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
'''
0) initialization
'''
date_start = 152
date_end = date_start+29
date_label = date_start+30
dir_source = '/groups/ESS/btang6/Projects/NOAA/Forecast_UFS/method3_xgboost/code/PM25_by_sample/2023/case1_train30days_predict1day/xls_files/'
filename_source = 'CONUS_US_interpolated_2023_'+str(date_start)+'_'+str(date_end)+'_PM25.xls'

'''
1) read-in prepared data
'''
def GetData(dir_input,filename_input):
    file_loc = dir_input
    obs_df = pd.read_excel(file_loc+filename_input)
    pd.set_option('display.max_columns',None)  
        
    TIME = list(obs_df['Time_UTC'])
    LOC_NUMBER = list(obs_df['Location_number'])

    EPA_OBS_PM25 = list(obs_df['EPA_OBS_PM25'])
    V1_BLH = list(obs_df['V1_BLH'])
    V2_D2M = list(obs_df['V2_D2M'])
    V3_E = list(obs_df['V3_E'])
    V4_SP = list(obs_df['V4_SP'])
    V5_T2M = list(obs_df['V5_T2M'])
    V6_TP = list(obs_df['V6_TP'])
    V7_U10 = list(obs_df['V7_U10'])
    V8_V10 = list(obs_df['V8_V10'])
    V9_AOD = list(obs_df['V9_AOD'])
    
    V11_LAND_USE_COVER = list(obs_df['V11_LAND_USE_COVER'])
    V12_ELEVATION = list(obs_df['V12_ELEVATION'])
    V13_POPULATION = list(obs_df['V13_POPULATION'])
    V14_UFS_AQM_PM25 = list(obs_df['V14_UFS_AQM_PM25'])
    
    V18_E_BC = list(obs_df['V18_E_BC'])
    V19_E_SO2 = list(obs_df['V19_E_SO2'])
    V20_E_NOX = list(obs_df['V20_E_NOX'])
    V21_E_VOC = list(obs_df['V21_E_VOC'])
    V22_E_NH3 = list(obs_df['V22_E_NH3'])
    V23_E_PM25 = list(obs_df['V23_E_PM25'])
  
    LAT = list(obs_df['LAT'])
    LON = list(obs_df['LON'])
    D1 = list(obs_df['D1'])
    D2 = list(obs_df['D2'])
    D3 = list(obs_df['D3'])
    D4 = list(obs_df['D4'])
    D5 = list(obs_df['D5'])
    JULIAN_DAY = list(obs_df['Julian_day'])
    
    
    return  V1_BLH, V2_D2M,V3_E, V4_SP, V5_T2M, V6_TP, V7_U10, V8_V10,V9_AOD, V11_LAND_USE_COVER,V12_ELEVATION,V13_POPULATION, V14_UFS_AQM_PM25,V18_E_BC, V19_E_SO2, V20_E_NOX, V21_E_VOC, V22_E_NH3, V23_E_PM25,EPA_OBS_PM25,LOC_NUMBER,LAT, LON, D1, D2, D3, D4, D5,JULIAN_DAY

V1_BLH, V2_D2M,V3_E, V4_SP, V5_T2M, V6_TP, V7_U10, V8_V10,V9_AOD, V11_LAND_USE_COVER,V12_ELEVATION,V13_POPULATION, V14_UFS_AQM_PM25,V18_E_BC, V19_E_SO2, V20_E_NOX, V21_E_VOC, V22_E_NH3, V23_E_PM25,EPA_OBS_PM25,LOC_NUMBER,LAT, LON, D1, D2, D3, D4, D5,JULIAN_DAY = GetData(dir_source,filename_source)


X1 = V1_BLH
X2 = V2_D2M
X3 = V3_E 
X4 = V4_SP
X5 = V5_T2M
X6 = V6_TP
X7 = V7_U10
X8 = V8_V10
X9 = V9_AOD

X11 = V11_LAND_USE_COVER
X12 = V12_ELEVATION
X13 = V13_POPULATION
X14 = V14_UFS_AQM_PM25

X18 = V18_E_BC
X19 = V19_E_SO2
X20 = V20_E_NOX
X21 = V21_E_VOC
X22 = V22_E_NH3
X23 = V23_E_PM25

X25 = LAT
X26 = LON
X27 = D1
X28 = D2
X29 = D3
X30 = D4
X31 = D5
X32 = JULIAN_DAY

Y =  EPA_OBS_PM25
LOC_NUMBER_ = LOC_NUMBER

'''
2) prepare input and output
'''
len_all = len(X1)
input_all= np.reshape(np.transpose([X1,X2,X3,X4,X5,X6,X7,X8,X9,X11,X12,X13,X14,X18,X19,X20,X21,X22,X23,X25,X26,X27,X28,X29,X30,X31,X32]),(len_all,27))

output_all = Y


'''
3-1) tunning each 10-fold model
'''
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint



max_depth = [1,3,5,7,9,10]
min_child_weight =[1,5,10]
gamma = [0.1, 0.2,0.5,0.8]
subsample = [0.5,1,2,3]
colsample_bytree=[0.5,1,2,3]
learning_rate = [0.01, 0.1, 0.15, 0.2]
n_estimators = [100,200,400,600, 800,1000]
random_grid = {'max_depth': max_depth,
               'min_child_weight': min_child_weight,
                'gamma': gamma,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators}
pprint(random_grid)


xgb_1 = xgb.XGBRegressor()
xgb_random_1 = RandomizedSearchCV(estimator = xgb_1, param_distributions = random_grid, n_iter = 300,scoring = 'r2', cv = 10)
xgb_random_1.fit(input_all, output_all)
print('ml_1\n',xgb_random_1.best_params_)
print(xgb_random_1.best_score_)


#print(xgb_random_1.best_params_['subsample'])
'''
5) save model 
'''

xgb_best_1 =xgb.XGBRegressor(subsample=xgb_random_1.best_params_['subsample'], 
                             n_estimators=xgb_random_1.best_params_['n_estimators'], 
                             min_child_weight=xgb_random_1.best_params_['min_child_weight'], 
                             max_depth=xgb_random_1.best_params_['max_depth'], 
                             learning_rate=xgb_random_1.best_params_['learning_rate'], 
                             gamma=xgb_random_1.best_params_['gamma'], 
                             colsample_bytree=xgb_random_1.best_params_['colsample_bytree'])

#to fit the model before save
xgb_best_1.fit(input_all, output_all)

import joblib
joblib.dump(xgb_best_1,'./xgb_model_'+str(date_label)+'.joblib')



















