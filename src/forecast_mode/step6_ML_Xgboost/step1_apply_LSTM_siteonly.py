'''
author: Beiming.Tang
date: 05/23/2025
'''
import numpy as np
import pandas as pd
from keras.models import load_model
from netCDF4 import Dataset
import joblib
from datetime import datetime as dt
from datetime import timedelta



'''
1) load LSTM model
'''
dir_lstm ='/data/aqf3/beiming.tang/DAFCOM/code/step4_ML_LSTM/IncludeJulianDay/Exclude_Spatial/Log_PM25/'
pattern_lstm = 'lstm_model_2025JantoApr_echo10batch48.h5'
lstm_model = load_model(dir_lstm+pattern_lstm)

Scaler_X = joblib.load(dir_lstm+'Scaler_X.save')
Scaler_Y = joblib.load(dir_lstm+'Scaler_Y.save')

pattern_csv = 'PM25_2025_01to04_HourlyData_LOG.csv'
df = pd.read_csv(dir_lstm+pattern_csv)
print('1 finish log mode')

'''
2) use only last 96 hours to predict
'''
df['time_utc'] = pd.to_datetime(df['time_utc'],format='%Y-%m-%d %H:%M:%S')
print('2 df shape',np.shape(df))

'''
3) get X and Y
'''
features = ['log_pm25',   #i do not need this in apply
            'v1_blh','v2_d2m','v3_e','v4_sp','v5_t2m','v6_tp','v7_u10','v8_v10','v9_aod','v10_luc',
            'v11_elevation','v12_population','v13_ufs_pm25','v14_e_bc','v15_e_nh3','v16_e_nox','v17_e_voc','v18_e_pm25','v19_e_so2',
            'hour_utc','day_of_year']
data_x = df[features[1:]]   #dim = (lines, 28)
data_scaled_X = Scaler_X.transform(data_x)

print('3 scaled X shape',np.shape(data_scaled_X))

'''
4) get sequence start lines
'''
time_step = 96

site_list = list(df['site_index'])
time_list = list(df['time_utc'])

squence_line_start_list = []
squence_site_list = []
squence_time_predict_list = []
for i in range(len(site_list)):
    if i+time_step <= len(site_list):
        if site_list[i+time_step-1] == site_list[i]:
            squence_line_start_list.append(i)
            squence_site_list.append(site_list[i])
            time_start = time_list[i]
            time_predict = time_start+timedelta(hours=time_step)
            squence_time_predict_list.append(time_predict)

print('4 len of squence start line',len(squence_line_start_list))

'''
5) create sequence
'''
X = []
for i in range(len(squence_line_start_list)):
    X.append(data_scaled_X[squence_line_start_list[i]:squence_line_start_list[i]+time_step,:])

X = np.array(X)
print('5 X shape',np.shape(X))

'''
6) make lstm predictions
'''
Y_pred = lstm_model.predict(X)                      #dim = (2617969, 24, 1)
Y_pred_2 = Y_pred.reshape(-1,1)                     #dim = (2617969*24, 1)
Y_original = Scaler_Y.inverse_transform(Y_pred_2)   #dim = (2617969*24, 1)
Y_pred_3 = Y_original.reshape(len(X),24,1)          #dim = (2617969,24,1)

print('6 Y pred shape',np.shape(Y_pred_3))

'''
7) write into csv
'''
Y_pred_1hr = Y_pred_3[:,0,:]   #dim = (2617969,1)
print('7 Y pred 1hr',np.shape(Y_pred_1hr))

df_out = pd.DataFrame(Y_pred_1hr,columns=['lstm_predict_log_pm25'])
df_out['time_utc'] = squence_time_predict_list
df_out['site_index'] = squence_site_list

df_out.to_csv('lstm_predict_siteonly_logPM.csv')
print('8 finish all works')





































