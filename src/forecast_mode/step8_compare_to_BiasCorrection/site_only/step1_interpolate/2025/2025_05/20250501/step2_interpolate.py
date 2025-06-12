'''
author: Beiming.Tang
date: 05/05/2025
'''
import numpy as np
import xlsxwriter
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd

'''
1) import model data
'''
from step1_model_inputs import LAT, LON, PM25_BC
from step1_model_inputs import start_date as start_date_model
from step1_model_inputs import date, dir_month

end_date_model   = start_date_model+timedelta(days=1)

'''
2) import obs data
'''
obs_init = '20250501_20250601'




dir_obs = '/data/aqf3/beiming.tang/DAFCOM/code/step2_obs_prepare/PM25/step1_get_airnow/2025/'+dir_month+'/'
pattern_obs = 'getobs_PM25_'+obs_init+'.xlsx'
df_obs = pd.read_excel(dir_obs+pattern_obs)

#Get narrow time range
df_obs['time_utc'] = pd.to_datetime(df_obs['time_utc'],format='%Y-%m-%d %H:%M:%S')
df_obs = df_obs[df_obs['time_utc'] >= start_date_model]
df_obs = df_obs[df_obs['time_utc'] < end_date_model]
df_obs = df_obs[df_obs['airnow_pm25'] >0]

#all obs after narrow time range
TIME_OBS = list(df_obs['time_utc'])
PM25_OBS = list(df_obs['airnow_pm25'])
LAT_OBS  = list(df_obs['lat'])
LON_OBS  = list(df_obs['lon'])
SITE_INDEX_OBS = list(df_obs['site_index'])

'''
3) define interpolation model
'''
def GetITP(time_input, lat_input, lon_input, variable_input):
    time_obs = time_input
    lat_obs = lat_input
    lon_obs = lon_input

    lat_model = LAT
    lon_model = LON
    variable_model = variable_input
    time_model_list = [start_date_model+timedelta(hours=X) for X in range(len(variable_model))]  #24

    c = 'no_value'
    for i in range(1,2400):
        if (lat_model[i-1] -lat_obs)*(lat_model[i]-lat_obs) <0:
            for j in range(1,6000):
                if (lon_model[j-1]-lon_obs)*(lon_model[j]-lon_obs) <0:
                    x1 = np.abs((lon_model[j-1]-lon_obs)/(lon_model[j-1]-lon_model[j]))
                    x2 = np.abs((lon_model[j]-lon_obs)/(lon_model[j-1]-lon_model[j]))
                    y1 = np.abs((lat_model[i-1]-lat_obs)/(lat_model[i-1]-lat_model[i]))
                    y2 = np.abs((lat_model[i]-lat_obs)/(lat_model[i-1]-lat_model[i]))
                    for k in range(len(variable_model)):
                        time_model_start_here = time_model_list[k]
                        time_model_end_here = time_model_start_here+timedelta(hours=1)
                        if time_model_start_here <= time_obs and time_model_end_here > time_obs:
                            a = x2 * variable_model[k][i-1][j-1] + x1 * variable_model[k][i-1][j]
                            b = x2 * variable_model[k][i][j-1] + x1 * variable_model[k][i][j]
                            c = y1 * b + y2 * a
                            break
                    break
            break
    return c

'''
4) 2D-INTERPOLATION MAIN
'''
V1 = PM25_BC

V1_final_list = []

for i in range(len(TIME_OBS)):
#for i in range(100):
    time_here = TIME_OBS[i]
    loc_number_here = SITE_INDEX_OBS[i]
    lat_here = LAT_OBS[i]
    lon_here = LON_OBS[i]
    
    V1_final = GetITP(time_here,lat_here,lon_here,V1)

    print(i,time_here,lat_here, lon_here, V1_final)
    V1_final_list.append(V1_final)

'''
5) write output into excel (xlsx format)
'''
#get a clean str time
TIME_OBS_str = []
for i in range(len(TIME_OBS)):
    time_obs_str = str(TIME_OBS[i])
    TIME_OBS_str.append(time_obs_str)

#form output list
my_list = [SITE_INDEX_OBS, TIME_OBS_str, LAT_OBS, LON_OBS,
           V1_final_list]        

#define output name
if date <10:
    pattern_output = pattern_obs.replace('getobs','CONUS').replace(obs_init,dir_month+'_0'+str(date))    
else:             
    pattern_output = pattern_obs.replace('getobs','CONUS').replace(obs_init,dir_month+'_'+str(date))

#all output 
workbook = xlsxwriter.Workbook(pattern_output)
worksheet = workbook.add_worksheet()
worksheet.write(0,0,'site_index')
worksheet.write(0,1,'time_utc')
worksheet.write(0,2,'lat')
worksheet.write(0,3,'lon')
worksheet.write(0,4,'bias_correction_pm25')

for row_num, row_data in enumerate(my_list):
    for col_num, data in enumerate(row_data):
        worksheet.write(col_num+1, row_num, data)
workbook.close()



































