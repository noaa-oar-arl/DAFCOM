'''
author: Beiming.Tang
date: 05/15/2025
'''
import numpy as np
import pandas as pd
from keras.models import load_model
from netCDF4 import Dataset
import joblib

'''
1) load LSTM model
'''
dir_lstm ='/data/aqf3/beiming.tang/DAFCOM/code/step4_ML_LSTM/IncludeJulianDay/Test_LSTM_configure/'
pattern_lstm = 'lstm_model_2025JantoApr_echo10batch48.h5'
lstm_model = load_model(dir_lstm+pattern_lstm)

print('finish load model')

Scaler_X = joblib.load('Scaler_X.save')
Scaler_Y = joblib.load('Scaler_Y.save')
print('finish section 1')


'''
2) load X inputs (1km all input, no need bias)
'''

from step1_prepare_X_input import BLH, D2M, E, SP, T2M, TP, U10, V10
from step1_prepare_X_input import AOD, Land_Use_Cover, Elevation, Population,UFS_AQM_PM25
from step1_prepare_X_input import E_BC_hourly, E_NH3_hourly, E_NOX_hourly, E_VOC_hourly, E_PM25_hourly, E_SO2_hourly
from step1_prepare_X_input import LAT_hourly ,LON_hourly, D1, D2,D3, D4, D5, hour_matrix, day_matrix

V1 = BLH
V2 = D2M
V3 = E
V4 = SP
V5 = T2M
V6 = TP
V7 = U10
V8 = V10
V9 = AOD
V10 = Land_Use_Cover
V11 = Elevation
V12 = Population
V13 = UFS_AQM_PM25
V14 = E_BC_hourly
V15 = E_NH3_hourly
V16 = E_NOX_hourly
V17 = E_VOC_hourly
V18 = E_PM25_hourly
V19 = E_SO2_hourly
V20 = LAT_hourly
V21 = LON_hourly
V22 = D1
V23 = D2
V24 = D3
V25 = D4
V26 = D5
V27 = hour_matrix
V28 = day_matrix
print('start section 2')
data_sum_1 = np.stack([V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,
                         V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,
                     V21,V22,V23,V24,V25,V26,V27,V28],axis=0) #dim = (28,96,2400,6000)
data_sum_2 = data_sum_1.transpose(1,2,3,0)                    #dim = (96,2400,6000,28)
data_sum_3 = data_sum_2.reshape(96,-1,28)                     #dim = (96,14400000,28)
data_sum_4 = data_sum_3.transpose(1,0,2)                      #dim = (14400000,96,28)
print('finish section 2')










'''
3) make prediction
'''
X_input = data_sum_4
Y_pred_1  = lstm_model.predict(X_input)         #dim = (1440000,24,1)
print('finish prediction')
Y_pred_2 = Y_pred_1.reshape(2400,6000,24)       #dim = (2400,6000,24)
Y_pred_3 = Y_pred_2.transpose(2,0,1)            #dim = (24,2400,6000)

print('finish section 3')

'''
4) save into netcdf
'''
ft = Dataset('LSTM_Bias_0p01.nc','w',format='NETCDF4')

time = ft.createDimension('time',None)
lat  = ft.createDimension('lat',2400)
lon  = ft.createDimension('lon',6000)

LAT = ft.createVariable('lat','f4',('lat','lon'))
LAT.units = ''
LAT.description = ''

LON = ft.createVariable('lon','f4',('lat','lon'))
LON.units = ''
LON.description = ''

LSTM_BIAS_PM25 = ft.createVariable('lstm_bias_pm25','f4',('time','lat','lon'))
LSTM_BIAS_PM25.units = ''
LSTM_BIAS_PM25.description = ''

LAT[:,:]= LAT
LON[:,:]= LON
LSTM_BIAS_PM25[:,:,:]= Y_pred_3 

ft.close()










