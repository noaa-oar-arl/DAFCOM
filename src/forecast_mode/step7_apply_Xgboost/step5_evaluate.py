# -*- coding: utf-8 -*-
"""
Created on Fri May 30 11:09:20 2025

@author: Beiming.Tang
"""

import numpy as np
import pandas as pd

dir_source = '/Users/beiming.tang/Desktop/1 year NOAA forecast run/code/code_xgboost_apply/'
pattern_csv = 'xgb_lstm_ufs_predictions.csv'

df = pd.read_csv(dir_source+pattern_csv)


'''
variables
'''
site_index = list(df['site_index'])
time = list(df['time'])
lat = list(df['lat'])
lon = list(df['lon'])

xgb_pm25 = list(df['xgb_predictions_pm25'])
lstm_pm25 = list(df['lstm_pm25'])
ufs_pm25 = list(df['ufs_pm25'])
obs_pm25 = list(df['obs_pm25'])

'''
evaluation
'''
import scipy
from sklearn.metrics import mean_squared_error

X1 = obs_pm25
X2 = xgb_pm25
X3 = lstm_pm25
X4 = ufs_pm25

slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(X1), np.array(X2))
mse_test = mean_squared_error(X1, X2)
mean_bias= np.mean(np.array(X2)-np.array(X1))
std_model = np.std(X2)
std_obs = np.std(X1)
print('this is Forecast XGB vs. OBS')
print('r test',r_value_test)
print('mse test',mse_test)
print('mean bias test', mean_bias)
print('std model =',std_model)
print('std obs =',std_obs)

print('/n')
print('/n')

slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(X1), np.array(X3))
mse_test = mean_squared_error(X1, X3)
mean_bias= np.mean(np.array(X3)-np.array(X1))
std_model = np.std(X3)
std_obs = np.std(X1)
print('this is Forecast LSTM vs. OBS')
print('r test',r_value_test)
print('mse test',mse_test)
print('mean bias test', mean_bias)
print('std model =',std_model)
print('std obs =',std_obs)

print('/n')
print('/n')

slope_test, intercept_test, r_value_test, p_value_test, std_err_test = scipy.stats.linregress(np.array(X1), np.array(X4))
mse_test = mean_squared_error(X1, X4)
mean_bias= np.mean(np.array(X4)-np.array(X1))
std_model = np.std(X4)
std_obs = np.std(X1)
print('this is Original UFS vs. OBS')
print('r test',r_value_test)
print('mse test',mse_test)
print('mean bias test', mean_bias)
print('std model =',std_model)
print('std obs =',std_obs)