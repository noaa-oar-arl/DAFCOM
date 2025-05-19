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
from step1_model_inputs import LAT, LON, BLH, D2M, E, SP, T2M, TP, U10, V10
from step1_model_inputs import AOD, Land_Use_Cover, Elevation, Population, UFS_AQM_PM25
from step1_model_inputs import E_BC_hourly, E_NH3_hourly, E_NOX_hourly, E_VOC_hourly, E_PM25_hourly, E_SO2_hourly
from step1_model_inputs import Lon_matrix_hourly, Lat_matrix_hourly, D1_matrix_hourly, D2_matrix_hourly, D3_matrix_hourly, D4_matrix_hourly, D5_matrix_hourly
from step1_model_inputs import start_date as start_date_model
from step1_model_inputs import date, dir_month

end_date_model   = start_date_model+timedelta(days=1)

'''
2) import obs data
'''
obs_init = '20250401_20250501'




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
            for j in range(6000):
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
V20 = Lon_matrix_hourly
V21 = Lat_matrix_hourly
V22 = D1_matrix_hourly
V23 = D2_matrix_hourly
V24 = D3_matrix_hourly
V25 = D4_matrix_hourly
V26 = D5_matrix_hourly

V1_final_list = []
V2_final_list = []
V3_final_list = []
V4_final_list = []
V5_final_list = []
V6_final_list = []
V7_final_list = []
V8_final_list = []
V9_final_list = []
V10_final_list = []
V11_final_list = []
V12_final_list = []
V13_final_list = []
V14_final_list = []
V15_final_list = []
V16_final_list = []
V17_final_list = []
V18_final_list = []
V19_final_list = []
V20_final_list = []
V21_final_list = []
V22_final_list = []
V23_final_list = []
V24_final_list = []
V25_final_list = []
V26_final_list = []

for i in range(len(TIME_OBS)):
#for i in range(100):
    time_here = TIME_OBS[i]
    loc_number_here = SITE_INDEX_OBS[i]
    lat_here = LAT_OBS[i]
    lon_here = LON_OBS[i]
    
    V1_final = GetITP(time_here,lat_here,lon_here,V1)
    V2_final = GetITP(time_here,lat_here,lon_here,V2)
    V3_final = GetITP(time_here,lat_here,lon_here,V3)
    V4_final = GetITP(time_here,lat_here,lon_here,V4)
    V5_final = GetITP(time_here,lat_here,lon_here,V5)
    V6_final = GetITP(time_here,lat_here,lon_here,V6)
    V7_final = GetITP(time_here,lat_here,lon_here,V7)
    V8_final = GetITP(time_here,lat_here,lon_here,V8)
    V9_final = GetITP(time_here,lat_here,lon_here,V9)
    V10_final = GetITP(time_here,lat_here,lon_here,V10)
    V11_final = GetITP(time_here,lat_here,lon_here,V11)
    V12_final = GetITP(time_here,lat_here,lon_here,V12)
    V13_final = GetITP(time_here,lat_here,lon_here,V13)
    V14_final = GetITP(time_here,lat_here,lon_here,V14)
    V15_final = GetITP(time_here,lat_here,lon_here,V15)
    V16_final = GetITP(time_here,lat_here,lon_here,V16)
    V17_final = GetITP(time_here,lat_here,lon_here,V17)
    V18_final = GetITP(time_here,lat_here,lon_here,V18)
    V19_final = GetITP(time_here,lat_here,lon_here,V19)
    V20_final = GetITP(time_here,lat_here,lon_here,V20)
    V21_final = GetITP(time_here,lat_here,lon_here,V21)
    V22_final = GetITP(time_here,lat_here,lon_here,V22)
    V23_final = GetITP(time_here,lat_here,lon_here,V23)
    V24_final = GetITP(time_here,lat_here,lon_here,V24)
    V25_final = GetITP(time_here,lat_here,lon_here,V25)
    V26_final = GetITP(time_here,lat_here,lon_here,V26)

    print(i,time_here,lat_here, lon_here, V1_final)
    V1_final_list.append(V1_final)
    V2_final_list.append(V2_final)
    V3_final_list.append(V3_final)
    V4_final_list.append(V4_final)
    V5_final_list.append(V5_final)
    V6_final_list.append(V6_final)
    V7_final_list.append(V7_final)
    V8_final_list.append(V8_final)
    V9_final_list.append(V9_final)
    V10_final_list.append(V10_final)
    V11_final_list.append(V11_final)
    V12_final_list.append(V12_final)
    V13_final_list.append(V13_final)
    V14_final_list.append(V14_final)
    V15_final_list.append(V15_final)
    V16_final_list.append(V16_final)
    V17_final_list.append(V17_final)
    V18_final_list.append(V18_final)
    V19_final_list.append(V19_final)
    V20_final_list.append(V20_final)
    V21_final_list.append(V21_final)
    V22_final_list.append(V22_final)
    V23_final_list.append(V23_final)
    V24_final_list.append(V24_final)
    V25_final_list.append(V25_final)
    V26_final_list.append(V26_final)

'''
5) write output into excel (xlsx format)
'''
#get a clean str time
TIME_OBS_str = []
for i in range(len(TIME_OBS)):
    time_obs_str = str(TIME_OBS[i])
    TIME_OBS_str.append(time_obs_str)

#form output list
my_list = [SITE_INDEX_OBS, TIME_OBS_str, LAT_OBS, LON_OBS, PM25_OBS,
           V1_final_list,  V2_final_list, V3_final_list, V4_final_list, V5_final_list,
           V6_final_list,  V7_final_list, V8_final_list, V9_final_list, V10_final_list,
           V11_final_list,  V12_final_list, V13_final_list, V14_final_list, V15_final_list,
           V16_final_list,  V17_final_list, V18_final_list, V19_final_list, V20_final_list,
           V21_final_list,  V22_final_list, V23_final_list, V24_final_list, V25_final_list,
           V26_final_list]        
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
worksheet.write(0,4,'airnow_obs_pm25')
worksheet.write(0,5,'v1_blh')
worksheet.write(0,6,'v2_d2m')
worksheet.write(0,7,'v3_e')
worksheet.write(0,8,'v4_sp')
worksheet.write(0,9,'v5_t2m')
worksheet.write(0,10,'v6_tp')
worksheet.write(0,11,'v7_u10')
worksheet.write(0,12,'v8_v10')
worksheet.write(0,13,'v9_aod')
worksheet.write(0,14,'v10_luc')
worksheet.write(0,15,'v11_elevation')
worksheet.write(0,16,'v12_population')
worksheet.write(0,17,'v13_ufs_pm25')
worksheet.write(0,18,'v14_e_bc')
worksheet.write(0,19,'v15_e_nh3')
worksheet.write(0,20,'v16_e_nox')
worksheet.write(0,21,'v17_e_voc')
worksheet.write(0,22,'v18_e_pm25')
worksheet.write(0,23,'v19_e_so2')
worksheet.write(0,24,'v20_lon')
worksheet.write(0,25,'v21_lat')
worksheet.write(0,26,'v22_d1')
worksheet.write(0,27,'v23_d2')
worksheet.write(0,28,'v24_d3')
worksheet.write(0,29,'v24_d4')
worksheet.write(0,30,'v25_d5')

for row_num, row_data in enumerate(my_list):
    for col_num, data in enumerate(row_data):
        worksheet.write(col_num+1, row_num, data)
workbook.close()



































