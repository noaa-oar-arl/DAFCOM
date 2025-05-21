'''
author: Beiming.Tang
date: 05/15/2025
'''
import numpy as np
import pandas as pd
from keras.models import load_model
from netCDF4 import Dataset
import joblib
from datetime import datetime as dt

'''
1) load LSTM model
'''
dir_lstm ='/data/aqf3/beiming.tang/DAFCOM/code/step4_ML_LSTM/IncludeJulianDay/Exclude_Spatial/Log_PM25/'
pattern_lstm = 'lstm_model_2025JantoApr_echo10batch48.h5'
lstm_model = load_model(dir_lstm+pattern_lstm)


Scaler_X = joblib.load('Scaler_X.save')
Scaler_Y = joblib.load('Scaler_Y.save')


dir_csv = '/data/aqf3/beiming.tang/DAFCOM/code/step4_ML_LSTM/IncludeJulianDay/Exclude_Spatial/Log_PM25/'
pattern_csv = 'PM25_2025JanToApr_HourlyData_LOG.csv'
df = pd.read_csv(dir_csv+pattern_csv)

'''
2) use only last 96 hours to predict
'''
df['time_utc'] = pd.to_datetime(df['time_utc'],format='%Y-%m-%d %H:%M:%S')

df_4days = df[(df['time_utc']>=dt(2025,4,27,0,0,0)) & (df['time_utc'] < dt(2025,5,1,0,0,0))]
print('df 4 days shape',np.shape(df_4days))

'''
3) get X and Y
'''
features = ['log_pm25',
            'v1_blh','v2_d2m','v3_e','v4_sp','v5_t2m','v6_tp','v7_u10','v8_v10','v9_aod','v10_luc',
            'v11_elevation','v12_population','v13_ufs_pm25','v14_e_bc','v15_e_nh3','v16_e_nox','v17_e_voc','v18_e_pm25','v19_e_so2',
            'hour_utc','day_of_year']
target = 'log_pm25'

data_x = df_4days[features[1:]]   #dim = (lines, 28)
data_scaled_X = Scaler_X.transform(data_x)

print('scaled X shape',np.shape(data_scaled_X))


'''
4) get sequence start lines
'''
time_step = 96

site_list = list(df_4days['site_index'])
time_list = list(df_4days['time_utc'])
lat_list  = list(df_4days['v21_lat'])
lon_list  = list(df_4days['v20_lon'])
obs_pm25_list = list(df_4days['airnow_obs_pm25'])

squence_line_start_list = []
squence_lat_list = []
squence_lon_list = []
squence_obs_pm25_list = []
for i in range(len(site_list)):
    if i+time_step <= len(site_list):
        if site_list[i+time_step-1] == site_list[i]:
            squence_line_start_list.append(i)
            squence_lat_list.append(lat_list[i])
            squence_lon_list.append(lon_list[i])
            squence_obs_pm25_list.append(obs_pm25_list[i])

print('len of squence start line',len(squence_line_start_list))

'''
5) create sequence
'''
X = []
for i in range(len(squence_line_start_list)):
    X.append(data_scaled_X[squence_line_start_list[i]:squence_line_start_list[i]+time_step,:])

X = np.array(X)
print('X shape',np.shape(X))

'''
6) make lstm predictions
'''
Y_pred = lstm_model.predict(X)   # dim = (331, 24, 1)
Y_pred_2 = Y_pred.reshape(-1,1)  # dim = (331*24, 1)
Y_original = Scaler_Y.inverse_transform(Y_pred_2)   #dim = (331*24, 1)
Y_pred_3 = Y_original.reshape(331,24,1)

print('Y pred shape',np.shape(Y_pred_3))


'''
7) write into csv
'''
Y_pred_avr = Y_pred_3.mean(axis=1).reshape(331,1)
df_out = pd.DataFrame(Y_pred_avr,columns=['pred_log_pm25'])

df_out['lat'] = squence_lat_list
df_out['lon'] = squence_lon_list
df_out['obs_pm25'] = squence_obs_pm25_list

df_out.to_csv('predict_0501_siteonly_logPM.csv')






































