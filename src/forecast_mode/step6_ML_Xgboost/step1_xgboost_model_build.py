'''
author: Beiming Tang
date: 05/14/2025
'''
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
dir_source = '/data/aqf3/beiming.tang/DAFCOM/data/PM25/'
filename_source = 'PM25_2025JanToApr_HourlyData.csv'

print('finish section 0')
'''
1) read-in prepared data
'''
def GetData(dir_input,filename_input):
    file_loc = dir_input
    obs_df = pd.read_csv(file_loc+filename_input)

    #TIME = list(obs_df['time_utc'])
    #LOC_NUMBER = list(obs_df['site_index'])
    EPA_OBS_PM25 = list(obs_df['airnow_obs_pm25'])

    V1_BLH = list(obs_df['v1_blh'])
    V2_D2M = list(obs_df['v2_d2m'])
    V3_E = list(obs_df['v3_e'])
    V4_SP = list(obs_df['v4_sp'])
    V5_T2M = list(obs_df['v5_t2m'])
    V6_TP = list(obs_df['v6_tp'])
    V7_U10 = list(obs_df['v7_u10'])
    V8_V10 = list(obs_df['v8_v10'])
    V9_AOD = list(obs_df['v9_aod'])
    V10_LAND_USE_COVER = list(obs_df['v10_luc'])
    V11_ELEVATION = list(obs_df['v11_elevation'])
    V12_POPULATION = list(obs_df['v12_population'])
    V13_UFS_AQM_PM25 = list(obs_df['v13_ufs_pm25'])
    V14_E_BC = list(obs_df['v14_e_bc'])
    V15_E_NH3 = list(obs_df['v15_e_nh3'])
    V16_E_NOX = list(obs_df['v16_e_nox'])
    V17_E_VOC = list(obs_df['v17_e_voc'])
    V18_E_PM25 = list(obs_df['v18_e_pm25'])
    V19_E_SO2 = list(obs_df['v19_e_so2'])
    LON = list(obs_df['v20_lon'])
    LAT = list(obs_df['v21_lat'])
    D1 = list(obs_df['v22_d1'])
    D2 = list(obs_df['v23_d2'])
    D3 = list(obs_df['v24_d3'])
    D4 = list(obs_df['v24_d4'])
    D5 = list(obs_df['v25_d5'])
    BIAS_PM25 = list(obs_df['bias_pm25'])
    HOUR_UTC = list(obs_df['hour_utc'])
    DAY_OF_YEAR = list(obs_df['day_of_year'])

    return  EPA_OBS_PM25,V1_BLH,V2_D2M,V3_E,V4_SP,V5_T2M,V6_TP,V7_U10,V8_V10,V9_AOD,V10_LAND_USE_COVER,V11_ELEVATION,V12_POPULATION,V13_UFS_AQM_PM25,V14_E_BC,V15_E_NH3,V16_E_NOX,V17_E_VOC,V18_E_PM25,V19_E_SO2,LON,LAT,D1,D2,D3,D4,D5,BIAS_PM25,HOUR_UTC,DAY_OF_YEAR

EPA_OBS_PM25,V1_BLH,V2_D2M,V3_E,V4_SP,V5_T2M,V6_TP,V7_U10,V8_V10,V9_AOD,V10_LAND_USE_COVER,V11_ELEVATION,V12_POPULATION,V13_UFS_AQM_PM25,V14_E_BC,V15_E_NH3,V16_E_NOX,V17_E_VOC,V18_E_PM25,V19_E_SO2,LON,LAT,D1,D2,D3,D4,D5,BIAS_PM25,HOUR_UTC,DAY_OF_YEAR= GetData(dir_source,filename_source)

X1 = V1_BLH
X2 = V2_D2M
X3 = V3_E
X4 = V4_SP
X5 = V5_T2M
X6 = V6_TP
X7 = V7_U10
X8 = V8_V10
X9 = V9_AOD
X10 = V10_LAND_USE_COVER
X11 = V11_ELEVATION
X12 = V12_POPULATION
X13 = V13_UFS_AQM_PM25
X14 = V14_E_BC
X15 = V15_E_NH3
X16 = V16_E_NOX
X17 = V17_E_VOC
X18 = V18_E_PM25
X19 = V19_E_SO2
X20 = LON
X21 = LAT
X22 = D1
X23 = D2
X24 = D3
X25 = D4
X26 = D5
X27 = BIAS_PM25
X28 = HOUR_UTC
X29 = DAY_OF_YEAR

Y =  EPA_OBS_PM25
#LOC_NUMBER_ = LOC_NUMBER
print('finish section 1')


'''
2) prepare input and output
'''
len_all = len(X1)
input_all= np.reshape(np.transpose([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29]),(len_all,29))

output_all = Y

print('finish section 2')


'''
3) tunning each 10-fold model
'''
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


print(xgb_random_1.best_params_['subsample'])
print('finih section 3')


'''
4) save model
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
joblib.dump(xgb_best_1,'./xgb_model_2025JanToApr.joblib')

print('finish section 4')









