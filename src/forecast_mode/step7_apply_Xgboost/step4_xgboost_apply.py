# -*- coding: utf-8 -*-
"""
Created on Fri May 30 10:15:22 2025

@author: Beiming.Tang
"""

import numpy as np
from netCDF4 import Dataset
import joblib
import sys
import xgboost as xgb
import pandas as pd

dir_model = '/Users/beiming.tang/Desktop/1 year NOAA forecast run/Data/ml_data/'
pattern_xgb = 'xgb_model_2025JanToApr.joblib'
model = joblib.load(dir_model+pattern_xgb)



pattern_csv = '3_merged_file_for_xgboost.csv'
df = pd.read_csv(dir_model+pattern_csv)

columns = list(df.columns)


df_features = df.drop(columns=['v20_lon','v21_lat','v22_d1','v23_d2','v24_d3','v24_d4','v25_d5',
                               'log_pm25','Unnamed: 0'])

columns_feature = list(df_features)

'''
re-order columns
'''
new_order = ['site_index','time_utc','lat','lon','airnow_obs_pm25',
             'v1_blh','v2_d2m', 'v3_e', 'v4_sp', 'v5_t2m', 
             'v6_tp', 'v7_u10', 'v8_v10','v9_aod', 'v10_luc', 
             'v11_elevation', 'v12_population', 'v13_ufs_pm25','v14_e_bc', 'v15_e_nh3', 
             'v16_e_nox', 'v17_e_voc', 'v18_e_pm25','v19_e_so2',
             'lstm_predict_log_pm25','hour_utc', 'day_of_year']
df_features_order = df_features[new_order]

columns_feature_order = list(df_features_order)

'''
prepare input
'''

X = df_features_order.iloc[:,5:]



with open('xgb_lstm_ufs_predictions.csv','w') as f:
    f.write('site_index,time,lat,lon,xgb_predictions_pm25,lstm_pm25,ufs_pm25,obs_pm25\n')
    for i in range(len(X)):
    # for i in range(5):
        row = X.iloc[i:i+1]
        xgb_pred = model.predict(row)[0]
        obs = df_features_order.iloc[i,4]
        lat = df_features_order.iloc[i,2]
        lon = df_features_order.iloc[i,3]
        time =  df_features_order.iloc[i,1]
        site_index = df_features_order.iloc[i,0]
        ufs_pm25 = df_features_order.iloc[i,17]
        lstm_pm25 = np.exp(df_features_order.iloc[i,24])    #covert log(pm)  to pm
        f.write(f"{site_index},{time},{lat},{lon},{xgb_pred},{lstm_pm25},{ufs_pm25},{obs}\n")