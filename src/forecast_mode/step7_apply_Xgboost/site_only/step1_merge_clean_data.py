'''
author: Beiming Tang
date: 05/20/2025
'''
import numpy as np
import xlsxwriter
import pandas as pd


'''
1) open data and merge
'''
year='2026'
pattern='May01to31'
dir_output='./'

dir_ = '/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Data/interpoalted_xlsx_files/'+year+'/'
day_list = ['05_01','05_02','05_03','05_04','05_05','05_06','05_07','05_08','05_09','05_10',
            '05_11','05_12','05_13','05_14','05_15','05_16','05_17','05_18','05_19','05_20',
            '05_21','05_22','05_23','05_24','05_25','05_26','05_27','05_28','05_29','05_30',
            '05_31']

df_list = []
for day in day_list:
    pattern_day = 'CONUS_PM25_'+year+'_'+day+'.xlsx'
    df_day = pd.read_excel(dir_+pattern_day)
    df_list.append(df_day)

df_total = pd.concat(df_list,ignore_index=True)

'''
2) drop nan values and duplicate values
'''
df_total =  df_total.dropna()
df_drop_nan  = df_total[df_total['v1_blh'] != 'no_value']
df_drop_duplicate = df_drop_nan.drop_duplicates(subset=['site_index','time_utc'],keep='first')

'''
3) sort by site_index and time_utc
'''
df_sort = df_drop_duplicate.sort_values(by=['site_index','time_utc']).reset_index(drop=True)

'''
4) add a column LOG(PM25)
'''
df_sort['log_pm25'] = np.log(df_sort['airnow_obs_pm25'])

'''
5) add day_of_year and hour (utc)
'''
time_utc = pd.to_datetime(df_sort['time_utc'])

hour_utc = time_utc.dt.hour
df_sort['hour_utc']= hour_utc

day_of_year = time_utc.dt.dayofyear
df_sort['day_of_year']=day_of_year

'''
6) save into a csv file
'''
#df_sort.to_excel('PM25_2025_JanToApr_HourlyData.xlsx',sheet_name='PM25',engine='xlsxwriter')

df_sort.to_csv(dir_output+'PM25_'+year+'_'+pattern+'_HourlyData_LOG.csv',index=False)





















































