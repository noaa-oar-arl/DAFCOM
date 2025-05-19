'''
author: Beimng Tang
date: 05/19/2025
this is a short copy of 
/data/aqf3/beiming.tang/DAFCOM/code/step4_ML_LSTM/IncludeJulianDay/Test_LSTM_configure/step2_LSTM_model_build.py
'''
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

'''
SECTION 1. sequence prepare
'''
'''
1-1) open csv
'''
dir_ = '/data/aqf3/beiming.tang/DAFCOM/code/step4_ML_LSTM/IncludeJulianDay/'
pattern = 'PM25_2025JanToApr_HourlyData.csv'
df = pd.read_csv(dir_+pattern)

'''
1-4) select features
'''
features = ['bias_pm25',
            'v1_blh','v2_d2m','v3_e','v4_sp','v5_t2m','v6_tp','v7_u10','v8_v10','v9_aod','v10_luc',
            'v11_elevation','v12_population','v13_ufs_pm25','v14_e_bc','v15_e_nh3','v16_e_nox','v17_e_voc','v18_e_pm25','v19_e_so2',
            'v20_lon','v21_lat','v22_d1','v23_d2','v24_d3','v24_d4','v25_d5','hour_utc','day_of_year']
target= 'bias_pm25'

data_x = df[features[1:]]
data_y =  df[features[0]]

#reshape data_y to make it 2D
data_y = pd.DataFrame(np.array(data_y).reshape((len(data_y),1)))
data_y = data_y.rename(columns={0:target})

'''
1-5) normalize data
'''
scaler_x = RobustScaler()
data_scaled_X = scaler_x.fit_transform(data_x)
print('x scaled shape',np.shape(data_scaled_X))

scaler_y = RobustScaler()
data_scaled_Y = scaler_y.fit_transform(data_y)
print('y scaled shape',np.shape(data_scaled_Y))


'''
2) save the scaler
'''
joblib.dump(scaler_x, 'Scaler_X.save')
joblib.dump(scaler_y, 'Scaler_Y.save')

#print(scaler_x)


