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
    file_loc = './'
    obs_df = pd.read_excel(file_loc+filename+'.xls')
    pd.set_option('display.max_columns',None)  
        
    TIME = list(obs_df['Time_UTC'])
    LOC_NUMBER = list(obs_df['Location_number'])

    EPA_PM25 = list(obs_df['EPA_OBS_PM25'])
    V1_BLH = list(obs_df['V1_BLH'])
    V2_D2M = list(obs_df['V2_D2M'])
    V3_E = list(obs_df['V3_E'])
    V4_SP = list(obs_df['V4_SP'])
    V5_T2M = list(obs_df['V5_T2M'])
    V6_TP = list(obs_df['V6_TP'])
    V7_U10 = list(obs_df['V7_U10'])
    V8_V10 = list(obs_df['V8_V10'])
    V9_AOD = list(obs_df['V9_AOD'])
    V10_NDVI = list(obs_df['V10_NDVI'])
    V11_LAND_USE_COVER = list(obs_df['V11_LAND_USE_COVER'])
    V12_ELEVATION = list(obs_df['V12_ELEVATION'])
    V13_POPULATION = list(obs_df['V13_POPULATION'])
    V14_CMAQ_PM25 = list(obs_df['V14_CMAQ_PM25'])
    # V15_CAMS_O3 = list(obs_df['V15_CAMS_O3'])
    # V16_CAMS_NO = list(obs_df['V16_CAMS_NO'])
    # V17_CAMS_NO2 = list(obs_df['V17_CAMS_NO2'])
    V18_E_BC = list(obs_df['V18_E_BC'])
    V19_E_SO2 = list(obs_df['V19_E_SO2'])
    V20_E_NOX = list(obs_df['V20_E_NOX'])
    V21_E_VOC = list(obs_df['V21_E_VOC'])
    V22_E_NH3 = list(obs_df['V22_E_NH3'])
    V23_E_PM25 = list(obs_df['V23_E_PM25'])
    # V24_FIRE_EMIS_PM25 = list(obs_df['V24_FIRE_EMIS_PM25'])
    LAT = list(obs_df['LAT'])
    LON = list(obs_df['LON'])
    D1 = list(obs_df['D1'])
    D2 = list(obs_df['D2'])
    D3 = list(obs_df['D3'])
    D4 = list(obs_df['D4'])
    D5 = list(obs_df['D5'])
    JULIAN_DAY = list(obs_df['Julian_day'])
    
    
    # return  V1_BLH, V2_D2M,V3_E, V4_SP, V5_T2M, V6_TP, V7_U10, V8_V10, V9_AOD, V10_NDVI, V11_LAND_USE_COVER,V12_ELEVATION,V13_POPULATION, V14_CAMS_PM25,V15_CAMS_O3,V16_CAMS_NO,V17_CAMS_NO2,V18_E_BC, V19_E_SO2, V20_E_NOX, V21_E_NMVOC, V22_E_NH3, V23_E_PM25, AirKorea_PM25,LOC_NUMBER,LAT, LON, D1, D2, D3, D4, D5,JULIAN_DAY
    return  V1_BLH, V2_D2M,V3_E, V4_SP, V5_T2M, V6_TP, V7_U10, V8_V10,V9_AOD,V10_NDVI, V11_LAND_USE_COVER,V12_ELEVATION,V13_POPULATION, V14_CMAQ_PM25,V18_E_BC, V19_E_SO2, V20_E_NOX, V21_E_VOC, V22_E_NH3, V23_E_PM25, EPA_PM25,LOC_NUMBER,LAT, LON, D1, D2, D3, D4, D5,JULIAN_DAY


V1_BLH, V2_D2M,V3_E, V4_SP, V5_T2M, V6_TP, V7_U10, V8_V10, V9_AOD, V10_NDVI, V11_LAND_USE_COVER,V12_ELEVATION,V13_POPULATION, V14_CMAQ_PM25,V18_E_BC, V19_E_SO2, V20_E_NOX, V21_E_VOC, V22_E_NH3, V23_E_PM25, EPA_PM25,LOC_NUMBER,LAT, LON, D1, D2, D3, D4, D5,JULIAN_DAY = GetData('CONUS_US_interpolated_June2023_fast')




X1 = V1_BLH
X2 = V2_D2M
X3 = V3_E 
X4 = V4_SP
X5 = V5_T2M
X6 = V6_TP
X7 = V7_U10
X8 = V8_V10
X9 = V9_AOD
X10 = V10_NDVI
X11 = V11_LAND_USE_COVER
X12 = V12_ELEVATION
X13 = V13_POPULATION
X14 = V14_CMAQ_PM25 


X18 = V18_E_BC
X19 = V19_E_SO2
X20 = V20_E_NOX
X21 = V21_E_VOC
X22 = V22_E_NH3
X23 = V23_E_PM25
# X24 = V24_FIRE_EMIS_PM25

X25 = LAT
X26 = LON
X27 = D1
X28 = D2
X29 = D3
X30 = D4
X31 = D5
X32 = JULIAN_DAY

Y =  EPA_PM25
LOC_NUMBER_ = LOC_NUMBER




'''
2) prepare input and output for 10-fold simulations
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

Index_list = [i for i in range(len(Y))]

Index_rest1, Index_part1 = train_test_split(Index_list, test_size =1/10)
Index_rest2, Index_part2 = train_test_split(Index_rest1, test_size =1/9)
Index_rest3, Index_part3 = train_test_split(Index_rest2, test_size =1/8)
Index_rest4, Index_part4 = train_test_split(Index_rest3, test_size =1/7)
Index_rest5, Index_part5 = train_test_split(Index_rest4, test_size =1/6)
Index_rest6, Index_part6 = train_test_split(Index_rest5, test_size =1/5)
Index_rest7, Index_part7 = train_test_split(Index_rest6, test_size =1/4)
Index_rest8, Index_part8 = train_test_split(Index_rest7, test_size =1/3)
Index_part9, Index_part10 = train_test_split(Index_rest8, test_size =1/2)


def InputAndOutput(Index_part1_input):
    X1_train = []
    X1_test = []
    X2_train = []
    X2_test = []
    X3_train = []
    X3_test = []
    X4_train = []
    X4_test = []
    X5_train = []
    X5_test = []
    X6_train = []
    X6_test = []
    X7_train = []
    X7_test = []
    X8_train = []
    X8_test = []
    X9_train = []
    X9_test = []
    X10_train = []
    X10_test = []
    X11_train = []
    X11_test = []
    X12_train = []
    X12_test = []
    X13_train = []
    X13_test = []
    X14_train = []
    X14_test = []
    
    X18_train = []
    X18_test = []
    X19_train = []
    X19_test = []
    X20_train = []
    X20_test = []
    X21_train = []
    X21_test = []
    X22_train = []
    X22_test = []
    X23_train = []
    X23_test = []
    # X24_train = []
    # X24_test = []
    
    X25_train = []
    X25_test = []
    X26_train = []
    X26_test = []
    X27_train = []
    X27_test = []
    X28_train = []
    X28_test = []
    X29_train = []
    X29_test = []
    X30_train = []
    X30_test = []
    X31_train = []
    X31_test = []
    X32_train = []
    X32_test = []
    
    Y_train = []
    Y_test = [] 
    
    for i in range(len(Y)):
        if i in Index_part1_input:
            X1_test.append(X1[i])
            X2_test.append(X2[i])
            X3_test.append(X3[i])
            X4_test.append(X4[i])
            X5_test.append(X5[i])
            X6_test.append(X6[i])
            X7_test.append(X7[i])
            X8_test.append(X8[i])
            X9_test.append(X9[i])
            X10_test.append(X10[i])
            X11_test.append(X11[i])
            X12_test.append(X12[i])
            X13_test.append(X13[i])
            X14_test.append(X14[i])
            X18_test.append(X18[i])
            X19_test.append(X19[i])
            X20_test.append(X20[i])
            X21_test.append(X21[i])
            X22_test.append(X22[i])
            X23_test.append(X23[i])
            # X24_test.append(X24[i])
            X25_test.append(X25[i])
            X26_test.append(X26[i])
            X27_test.append(X27[i])
            X28_test.append(X28[i])
            X29_test.append(X29[i])
            X30_test.append(X30[i])
            X31_test.append(X31[i])
            X32_test.append(X32[i])
            Y_test.append(Y[i])            
        
        else:
            X1_train.append(X1[i])
            X2_train.append(X2[i])
            X3_train.append(X3[i])
            X4_train.append(X4[i])
            X5_train.append(X5[i])
            X6_train.append(X6[i])
            X7_train.append(X7[i])
            X8_train.append(X8[i])
            X9_train.append(X9[i])
            X10_train.append(X10[i])
            X11_train.append(X11[i])
            X12_train.append(X12[i])
            X13_train.append(X13[i])
            X14_train.append(X14[i])
            X18_train.append(X18[i])
            X19_train.append(X19[i])
            X20_train.append(X20[i])
            X21_train.append(X21[i])
            X22_train.append(X22[i])
            X23_train.append(X23[i])
            # X24_train.append(X24[i])
            X25_train.append(X25[i])
            X26_train.append(X26[i])
            X27_train.append(X27[i])
            X28_train.append(X28[i])
            X29_train.append(X29[i])
            X30_train.append(X30[i])
            X31_train.append(X31[i])
            X32_train.append(X32[i])
            Y_train.append(Y[i])
    
    len_test = len(X1_test)
    len_train = len(X1_train)
    
    input_train = np.reshape(np.transpose([X1_train,X2_train,X3_train,X4_train,X5_train,X6_train,X7_train,X8_train,X9_train,X10_train,X11_train,X12_train,X13_train,X14_train,X18_train,X19_train,X20_train,X21_train,X22_train,X23_train,X25_train,X26_train,X27_train,X28_train,X29_train,X30_train,X31_train,X32_train]),(len_train,28))
    output_train = Y_train
    input_test = np.reshape(np.transpose([X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test,X8_test, X9_test,X10_test,X11_test,X12_test,X13_test,X14_test,X18_test,X19_test,X20_test,X21_test,X22_test,X23_test,X25_test,X26_test,X27_test,X28_test,X29_test,X30_test,X31_test,X32_test]),(len_test,28))
    output_test = Y_test
    
    return input_train,output_train,input_test,output_test
    

input_train_loc1,output_train_loc1,input_test_loc1,output_test_loc1 = InputAndOutput(Index_part1)
input_train_loc2,output_train_loc2,input_test_loc2,output_test_loc2 = InputAndOutput(Index_part2)
input_train_loc3,output_train_loc3,input_test_loc3,output_test_loc3 = InputAndOutput(Index_part3)
input_train_loc4,output_train_loc4,input_test_loc4,output_test_loc4 = InputAndOutput(Index_part4)
input_train_loc5,output_train_loc5,input_test_loc5,output_test_loc5 = InputAndOutput(Index_part5)
input_train_loc6,output_train_loc6,input_test_loc6,output_test_loc6 = InputAndOutput(Index_part6)
input_train_loc7,output_train_loc7,input_test_loc7,output_test_loc7 = InputAndOutput(Index_part7)
input_train_loc8,output_train_loc8,input_test_loc8,output_test_loc8 = InputAndOutput(Index_part8)
input_train_loc9,output_train_loc9,input_test_loc9,output_test_loc9 = InputAndOutput(Index_part9)
input_train_loc10,output_train_loc10,input_test_loc10,output_test_loc10 = InputAndOutput(Index_part10)            



       
'''
3) evaluate
'''

'''
3-1) tunning each 10-fold model
'''
# import scipy 
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import RandomizedSearchCV
# from pprint import pprint

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] 
# max_features = ['auto', 'sqrt']                                                
# max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]                  
# max_depth.append(None)
# min_samples_split = [2, 5, 10]                                                 
# min_samples_leaf = [1, 2, 4]                                                   
# bootstrap = [True, False]                                                      

# random_grid = {'n_estimators': n_estimators,
#                 'max_features': max_features,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'min_samples_leaf': min_samples_leaf,
#                 'bootstrap': bootstrap}
# pprint(random_grid)

# rf_1 = RandomForestRegressor()                                                  
# rf_random_1 = RandomizedSearchCV(estimator = rf_1, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_1.fit(input_train_loc1, output_train_loc1)
# print(rf_random_1.best_params_)
# # result rf_1 {'n_estimators': 600, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 70, 'bootstrap': False}

# rf_2 = RandomForestRegressor()                                                  
# rf_random_2 = RandomizedSearchCV(estimator = rf_2, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_2.fit(input_train_loc2, output_train_loc2)
# print(rf_random_2.best_params_)
# #result rf_2 {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 90, 'bootstrap': False}

# rf_3 = RandomForestRegressor()                                                  
# rf_random_3 = RandomizedSearchCV(estimator = rf_3, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_3.fit(input_train_loc3, output_train_loc3)
# print(rf_random_3.best_params_)
# #result rf_3 {'n_estimators': 1800, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}

# rf_4 = RandomForestRegressor()                                                  
# rf_random_4 = RandomizedSearchCV(estimator = rf_4, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_4.fit(input_train_loc4, output_train_loc4)
# print('4',rf_random_4.best_params_)
# #result rf_4 {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}

# rf_5 = RandomForestRegressor()                                                  
# rf_random_5 = RandomizedSearchCV(estimator = rf_5, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_5.fit(input_train_loc5, output_train_loc5)
# print('5',rf_random_5.best_params_)
# #result rf_5 {'n_estimators': 1400, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': False}

# rf_6 = RandomForestRegressor()                                                  
# rf_random_6 = RandomizedSearchCV(estimator = rf_6, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_6.fit(input_train_loc6, output_train_loc6)
# print('6',rf_random_6.best_params_)
# #result rf_6 

# rf_7 = RandomForestRegressor()                                                  
# rf_random_7 = RandomizedSearchCV(estimator = rf_7, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_7.fit(input_train_loc7, output_train_loc7)
# print('7',rf_random_7.best_params_)
# #result rf_7

# rf_8 = RandomForestRegressor()                                                  
# rf_random_8 = RandomizedSearchCV(estimator = rf_8, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_8.fit(input_train_loc8, output_train_loc8)
# print('8',rf_random_8.best_params_)
# #result rf_8 

# rf_9 = RandomForestRegressor()                                                  
# rf_random_9 = RandomizedSearchCV(estimator = rf_9, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_9.fit(input_train_loc9, output_train_loc9)
# print('9',rf_random_9.best_params_)
# #result rf_9 

# rf_10 = RandomForestRegressor()                                                  
# rf_random_10 = RandomizedSearchCV(estimator = rf_10, param_distributions = random_grid, n_iter = 100,scoring = 'r2', cv = 10)                                                                                                                                                        
# rf_random_10.fit(input_train_loc10, output_train_loc10)
# print('10',rf_random_10.best_params_)
# #result rf_10 




'''
3-2)make prediction
'''
rf_best_1 =RandomForestRegressor(n_estimators = 200, min_samples_split = 2, min_samples_leaf =2, max_features = 'sqrt', max_depth = 80, bootstrap = False) 
rf_best_1.fit(input_train_loc1,output_train_loc1)
Y_hat_test1 = rf_best_1.predict(input_test_loc1)

rf_best_2 =RandomForestRegressor(n_estimators = 2000, min_samples_split = 2, min_samples_leaf =2, max_features = 'sqrt', max_depth = 40, bootstrap = False) 
rf_best_2.fit(input_train_loc2,output_train_loc2)
Y_hat_test2 = rf_best_2.predict(input_test_loc2)

rf_best_3 =RandomForestRegressor(n_estimators = 1600, min_samples_split = 5, min_samples_leaf =2, max_features = 'sqrt', max_depth = 30, bootstrap = False) 
rf_best_3.fit(input_train_loc3,output_train_loc3)
Y_hat_test3 = rf_best_3.predict(input_test_loc3)

rf_best_4 =RandomForestRegressor(n_estimators = 400, min_samples_split = 2, min_samples_leaf =1, max_features = 'sqrt', max_depth = 90, bootstrap = False) 
rf_best_4.fit(input_train_loc4,output_train_loc4)
Y_hat_test4 = rf_best_4.predict(input_test_loc4)

rf_best_5 =RandomForestRegressor(n_estimators = 200, min_samples_split = 2, min_samples_leaf =2, max_features = 'sqrt', max_depth = 50, bootstrap = False) 
rf_best_5.fit(input_train_loc5,output_train_loc5)
Y_hat_test5 = rf_best_5.predict(input_test_loc5)

rf_best_6 =RandomForestRegressor(n_estimators = 800, min_samples_split = 5, min_samples_leaf =2, max_features = 'sqrt', max_depth = 30, bootstrap = False) 
rf_best_6.fit(input_train_loc6,output_train_loc6)
Y_hat_test6 = rf_best_6.predict(input_test_loc6)

rf_best_7 =RandomForestRegressor(n_estimators = 1600, min_samples_split = 5, min_samples_leaf =2, max_features = 'sqrt', max_depth = 90, bootstrap = False) 
rf_best_7.fit(input_train_loc7,output_train_loc7)
Y_hat_test7 = rf_best_7.predict(input_test_loc7)

rf_best_8 =RandomForestRegressor(n_estimators = 1600, min_samples_split = 5, min_samples_leaf =2, max_features = 'sqrt', max_depth = 80, bootstrap = False) 
rf_best_8.fit(input_train_loc8,output_train_loc8)
Y_hat_test8 = rf_best_8.predict(input_test_loc8)

rf_best_9 =RandomForestRegressor(n_estimators = 800, min_samples_split = 2, min_samples_leaf =1, max_features = 'sqrt', max_depth = 70, bootstrap = False) 
rf_best_9.fit(input_train_loc9,output_train_loc9)
Y_hat_test9 = rf_best_9.predict(input_test_loc9)

rf_best_10 =RandomForestRegressor(n_estimators = 400, min_samples_split = 2, min_samples_leaf =2, max_features = 'sqrt', max_depth = 80, bootstrap = False) 
rf_best_10.fit(input_train_loc10,output_train_loc10)
Y_hat_test10 = rf_best_10.predict(input_test_loc10)




'''
3-3) sum list for all 10% test result
'''

Y_hat_test_all = []
for i in range(len(Y_hat_test1)):
    Y_hat_test_all.append(Y_hat_test1[i])
    
for i in range(len(Y_hat_test2)):
    Y_hat_test_all.append(Y_hat_test2[i])
    
for i in range(len(Y_hat_test3)):
    Y_hat_test_all.append(Y_hat_test3[i])
    
for i in range(len(Y_hat_test4)):
    Y_hat_test_all.append(Y_hat_test4[i])
    
for i in range(len(Y_hat_test5)):
    Y_hat_test_all.append(Y_hat_test5[i])
    
for i in range(len(Y_hat_test6)):
    Y_hat_test_all.append(Y_hat_test6[i])
    
for i in range(len(Y_hat_test7)):
    Y_hat_test_all.append(Y_hat_test7[i])
    
for i in range(len(Y_hat_test8)):
    Y_hat_test_all.append(Y_hat_test8[i])
    
for i in range(len(Y_hat_test9)):
    Y_hat_test_all.append(Y_hat_test9[i])
    
for i in range(len(Y_hat_test10)):
    Y_hat_test_all.append(Y_hat_test10[i])



Y_test_all = []
for i in range(len(output_test_loc1)):
    Y_test_all.append(output_test_loc1[i])
    
for i in range(len(output_test_loc2)):
    Y_test_all.append(output_test_loc2[i])
    
for i in range(len(output_test_loc3)):
    Y_test_all.append(output_test_loc3[i])
    
for i in range(len(output_test_loc4)):
    Y_test_all.append(output_test_loc4[i])

for i in range(len(output_test_loc5)):
    Y_test_all.append(output_test_loc5[i])
    
for i in range(len(output_test_loc6)):
    Y_test_all.append(output_test_loc6[i])

for i in range(len(output_test_loc7)):
    Y_test_all.append(output_test_loc7[i])
    
for i in range(len(output_test_loc8)):
    Y_test_all.append(output_test_loc8[i])

for i in range(len(output_test_loc9)):
    Y_test_all.append(output_test_loc9[i])
    
for i in range(len(output_test_loc10)):
    Y_test_all.append(output_test_loc10[i])    

'''
4) final evaluation
'''
import scipy 
from sklearn.metrics import mean_squared_error   


slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(Y_hat_test_all), np.array(Y_test_all))
rms_test = mean_squared_error(Y_test_all, Y_hat_test_all, squared=False)
mean_bias_test = np.mean(np.array(Y_hat_test_all)-np.array(Y_test_all))
print('r test',r_value_test)
# print('std test', std_err_test)
print('rms test',rms_test)
print('mean bias test', mean_bias_test)

std_model = np.std(Y_hat_test_all)
std_obs = np.std(Y_test_all)

print('std model =',std_model)
print('std obs =',std_obs)





'''
5) save model 
'''

import joblib
# joblib.dump(rf_best_1,'./random_forest_May2021_method4_10fold_model1.joblib')
# joblib.dump(rf_best_2,'./random_forest_May2021_method4_10fold_model2.joblib')
# joblib.dump(rf_best_3,'./random_forest_May2021_method4_10fold_model3.joblib')
# joblib.dump(rf_best_4,'./random_forest_May2021_method4_10fold_model4.joblib')
# joblib.dump(rf_best_5,'./random_forest_May2021_method4_10fold_model5.joblib')
# joblib.dump(rf_best_6,'./random_forest_May2021_method4_10fold_model6.joblib')
# joblib.dump(rf_best_7,'./random_forest_May2021_method4_10fold_model7.joblib')
# joblib.dump(rf_best_8,'./random_forest_May2021_method4_10fold_model8.joblib')
joblib.dump(rf_best_9,'./random_forest_May2021_method4_10fold_model9.joblib')
# joblib.dump(rf_best_10,'./random_forest_May2021_method4_10fold_model10.joblib')


'''
6) get importance score
'''
# Get numerical feature importances
importances = list(rf_best_1.feature_importances_)
# List of tuples with variable and importance
feature_list = ['V1_BLH','V2_D2M','V3_E','V4_SP','V5_T2M','V6_TP','V7_U10','V8_V10','V9_AOD','V10_NDVI','V11_LAND_USE_COVER','V12_ELEVATION','V13_POPULATION',
                'V14_CTM_PM25','V18_E_BC','V19_E_SO2','V20_E_NOX','V21_E_VOC','V22_E_NH3','V23_E_PM25','LAT','LON','D1','D2','D3','D4','D5','Julian_day(time)']
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


'''
7) pick 1 model from 10
'''

def ChooseModel(rf_best_input):

    Y_hat_test1 = rf_best_input.predict(input_test_loc1)
    Y_hat_test2 = rf_best_input.predict(input_test_loc2)
    Y_hat_test3 = rf_best_input.predict(input_test_loc3)
    Y_hat_test4 = rf_best_input.predict(input_test_loc4)
    Y_hat_test5 = rf_best_input.predict(input_test_loc5)
    Y_hat_test6 = rf_best_input.predict(input_test_loc6)
    Y_hat_test7 = rf_best_input.predict(input_test_loc7)
    Y_hat_test8 = rf_best_input.predict(input_test_loc8)
    Y_hat_test9 = rf_best_input.predict(input_test_loc9)
    Y_hat_test10 = rf_best_input.predict(input_test_loc10)

    Y_hat_test_all = []
    for i in range(len(Y_hat_test1)):
        Y_hat_test_all.append(Y_hat_test1[i])    
    for i in range(len(Y_hat_test2)):
        Y_hat_test_all.append(Y_hat_test2[i])    
    for i in range(len(Y_hat_test3)):
        Y_hat_test_all.append(Y_hat_test3[i])    
    for i in range(len(Y_hat_test4)):
        Y_hat_test_all.append(Y_hat_test4[i])    
    for i in range(len(Y_hat_test5)):
        Y_hat_test_all.append(Y_hat_test5[i])    
    for i in range(len(Y_hat_test6)):
        Y_hat_test_all.append(Y_hat_test6[i])    
    for i in range(len(Y_hat_test7)):
        Y_hat_test_all.append(Y_hat_test7[i])    
    for i in range(len(Y_hat_test8)):
        Y_hat_test_all.append(Y_hat_test8[i])   
    for i in range(len(Y_hat_test9)):
        Y_hat_test_all.append(Y_hat_test9[i])   
    for i in range(len(Y_hat_test10)):
        Y_hat_test_all.append(Y_hat_test10[i])
    
    
    
    Y_test_all = []
    for i in range(len(output_test_loc1)):
        Y_test_all.append(output_test_loc1[i])   
    for i in range(len(output_test_loc2)):
        Y_test_all.append(output_test_loc2[i])    
    for i in range(len(output_test_loc3)):
        Y_test_all.append(output_test_loc3[i])    
    for i in range(len(output_test_loc4)):
        Y_test_all.append(output_test_loc4[i])
    for i in range(len(output_test_loc5)):
        Y_test_all.append(output_test_loc5[i])   
    for i in range(len(output_test_loc6)):
        Y_test_all.append(output_test_loc6[i])
    for i in range(len(output_test_loc7)):
        Y_test_all.append(output_test_loc7[i])    
    for i in range(len(output_test_loc8)):
        Y_test_all.append(output_test_loc8[i])
    for i in range(len(output_test_loc9)):
        Y_test_all.append(output_test_loc9[i])    
    for i in range(len(output_test_loc10)):
        Y_test_all.append(output_test_loc10[i]) 

    slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(Y_hat_test_all), np.array(Y_test_all))
    rms_test = mean_squared_error(Y_test_all, Y_hat_test_all, squared=False)
    mean_bias_test = np.mean(np.array(Y_hat_test_all)-np.array(Y_test_all))
    
    return r_value_test, std_err_test,rms_test,mean_bias_test


r_value_test_model1, std_err_test_model1,rms_test_model1,mean_bias_test_model1 = ChooseModel(rf_best_1)
r_value_test_model2, std_err_test_model2,rms_test_model2,mean_bias_test_model2 = ChooseModel(rf_best_2)
r_value_test_model3, std_err_test_model3,rms_test_model3,mean_bias_test_model3 = ChooseModel(rf_best_3)
r_value_test_model4, std_err_test_model4,rms_test_model4,mean_bias_test_model4 = ChooseModel(rf_best_4)
r_value_test_model5, std_err_test_model5,rms_test_model5,mean_bias_test_model5 = ChooseModel(rf_best_5)
r_value_test_model6, std_err_test_model6,rms_test_model6,mean_bias_test_model6 = ChooseModel(rf_best_6)
r_value_test_model7, std_err_test_model7,rms_test_model7,mean_bias_test_model7 = ChooseModel(rf_best_7)
r_value_test_model8, std_err_test_model8,rms_test_model8,mean_bias_test_model8 = ChooseModel(rf_best_8)
r_value_test_model9, std_err_test_model9,rms_test_model9,mean_bias_test_model9 = ChooseModel(rf_best_9)
r_value_test_model10, std_err_test_model10,rms_test_model10,mean_bias_test_model10 = ChooseModel(rf_best_10)

print('1',r_value_test_model1, std_err_test_model1,rms_test_model1,mean_bias_test_model1)
print('2',r_value_test_model2, std_err_test_model2,rms_test_model2,mean_bias_test_model2)
print('3',r_value_test_model3, std_err_test_model3,rms_test_model3,mean_bias_test_model3)
print('4',r_value_test_model4, std_err_test_model4,rms_test_model4,mean_bias_test_model4)
print('5',r_value_test_model5, std_err_test_model5,rms_test_model5,mean_bias_test_model5)
print('6',r_value_test_model6, std_err_test_model6,rms_test_model6,mean_bias_test_model6)
print('7',r_value_test_model7, std_err_test_model7,rms_test_model7,mean_bias_test_model7)
print('8',r_value_test_model8, std_err_test_model8,rms_test_model8,mean_bias_test_model8)
print('9',r_value_test_model9, std_err_test_model9,rms_test_model9,mean_bias_test_model9)
print('10',r_value_test_model10, std_err_test_model10,rms_test_model10,mean_bias_test_model10)



















