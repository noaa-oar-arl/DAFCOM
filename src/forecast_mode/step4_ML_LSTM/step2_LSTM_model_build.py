'''
author: Beiming.Tang
date: 05/07/2025
'''
import numpy as np
import xlsxwriter
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd

'''
SECTION 1. sequence prepare
'''
'''
1-1) open csv
'''
dir_ = '/data/aqf3/beiming.tang/DAFCOM/code/step4_ML_LSTM/IncludeJulianDay/Exclude_Spatial/Log_PM25/'
pattern = 'PM25_2025JanToApr_HourlyData_LOG.csv'
df = pd.read_csv(dir_+pattern)

'''
1-2) convert time to datetime
'''
df['time_utc'] = pd.to_datetime(df['time_utc'],format='%Y-%m-%d %H:%M:%S')
#print(type(df['time_utc'][100]))


'''
1-3) get sequence start line number
'''
time_step = 120
time_step_short = 96  #use 96 hrs to forecast 24 hrs. total 120 hrs

site_list = list(df['site_index'])
time_list = list(df['time_utc'])

squence_line_start_list = []
for i in range(len(site_list)):
    if i+time_step <= len(site_list):
        if site_list[i+time_step-1] == site_list[i]:
            # print(i)
            time_start = time_list[i]
            time_end = time_list[i+time_step-1]
            if time_start+timedelta(hours=time_step) == time_end:
                squence_line_start_list.append(i)
                
print(len(squence_line_start_list))

'''
1-4) select features
'''
features = ['log_pm25',
            'v1_blh','v2_d2m','v3_e','v4_sp','v5_t2m','v6_tp','v7_u10','v8_v10','v9_aod','v10_luc',
            'v11_elevation','v12_population','v13_ufs_pm25','v14_e_bc','v15_e_nh3','v16_e_nox','v17_e_voc','v18_e_pm25','v19_e_so2',
            'hour_utc','day_of_year']
target= 'log_pm25'

data_x = df[features[1:]]
data_y =  df[features[0]]

#reshape data_y to make it 2D
data_y = pd.DataFrame(np.array(data_y).reshape((len(data_y),1)))
data_y = data_y.rename(columns={0:target})

'''
1-5) normalize data
'''
from sklearn.preprocessing import RobustScaler
import joblib

scaler_x = RobustScaler()
data_scaled_X = scaler_x.fit_transform(data_x)

scaler_y = RobustScaler()
data_scaled_Y = scaler_y.fit_transform(data_y)


joblib.dump(scaler_x, 'Scaler_X.save')
joblib.dump(scaler_y, 'Scaler_Y.save')

'''
1-6) create sequence
'''
X = []
Y = []
for i in range(len(squence_line_start_list)):
    X.append(data_scaled_X[squence_line_start_list[i]:squence_line_start_list[i]+time_step_short,:])
    Y.append(data_scaled_Y[squence_line_start_list[i]+time_step_short:squence_line_start_list[i]+time_step,:])

X = np.array(X)
Y = np.array(Y)

print(np.shape(X))
print(np.shape(Y))

'''
1-7) split into train and test
'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


'''
1-8) reshape dimension for LSTM input format
'''


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

print(f"Reshaped Train: {X_train.shape}, Reshaped Test: {X_test.shape}")
print('this is the end of section 1')





'''
SECTION 2. build LSTM model
'''
'''
2-1) build LSTM model
'''
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense
# Define LSTM model
model = Sequential([
    LSTM(units=84, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),dropout=0.2),  # First LSTM layer
    LSTM(units=84, return_sequences=False,dropout=0.2),  # Second LSTM layer
    Dense(units=28),  # Fully connected layer
    Dense(units=24)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
model.summary()

'''
2-2) train the model
'''
# Train model
history = model.fit(X_train, Y_train, epochs=10, batch_size=48, validation_data=(X_test, Y_test))

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("LSTM Training Loss")
#plt.show()
plt.savefig('f1.LSTM_training_loss_echo10batch24.png')


# Save the trained model
model.save("lstm_model_2025JantoApr_echo10batch48.h5")

'''
SECTION 2. save into csv
'''
Y_predict = model.predict(X_test)

print(np.shape(Y_predict))
print(np.shape(Y_test))

#reverse scale
Y_test = Y_test.reshape(np.shape(Y_test[:,:,0]))
Y_predict_reverseScale = scaler_y.inverse_transform(Y_predict)
Y_test_reverseScale = scaler_y.inverse_transform(Y_test)

#save to csv
pd.DataFrame(Y_predict_reverseScale).to_csv('Y_predict_original.csv',index=False)
pd.DataFrame(Y_test_reverseScale).to_csv('Y_test_original.csv',index=False)

print('this is end of section 2')















