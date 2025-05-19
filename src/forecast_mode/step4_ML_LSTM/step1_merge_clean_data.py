'''
author: Beiming Tang
date: 05/07/2025
'''
import numpy as np
import xlsxwriter
import pandas as pd


'''
1) open data and merge
'''
dir_ = '/data/aqf3/beiming.tang/DAFCOM/data/PM25/interpolate_xlsx_files/'

day_list = ['01_01','01_02','01_03','01_04','01_05','01_06','01_07','01_08','01_09','01_10',
            '01_11','01_12','01_13','01_14','01_15','01_16','01_17','01_18','01_19','01_20',
            '01_21','01_22','01_23','01_24','01_25','01_26','01_27','01_28','01_29','01_30','01_31',
            '02_01','02_02','02_03','02_04','02_05','02_06','02_07','02_08','02_09','02_10',
            '02_11','02_12','02_13','02_14','02_15','02_16','02_17','02_18','02_19','02_20',
            '02_21','02_22','02_23','02_24','02_25','02_26','02_27','02_28',
            '03_01','03_02','03_03','03_04','03_05','03_06','03_07','03_08','03_09','03_10',
            '03_11','03_12','03_13','03_14','03_15','03_16','03_17','03_18','03_19','03_20',
            '03_21','03_22','03_23','03_24','03_25','03_26','03_27','03_28','03_29','03_30','03_31',
            '04_01','04_02','04_03','04_04','04_05','04_06','04_07','04_08','04_09','04_10',
            '04_11','04_12','04_13','04_14','04_15','04_16','04_17','04_18','04_19','04_20',
            '04_21','04_22','04_23','04_24','04_25','04_26','04_27','04_28','04_29','04_30'
            ]

df_list = []
for day in day_list:
    pattern_day = 'CONUS_PM25_2025_'+day+'.xlsx'
    df_day = pd.read_excel(dir_+pattern_day)
    df_list.append(df_day)

df_total = pd.concat(df_list,ignore_index=True)

'''
2) drop nan values and duplicate values
'''
df_drop_nan  = df_total[df_total['v1_blh'] != 'no_value']
df_drop_duplicate = df_drop_nan.drop_duplicates(subset=['site_index','time_utc'],keep='first')

'''
3) sort by site_index and time_utc
'''
df_sort = df_drop_duplicate.sort_values(by=['site_index','time_utc']).reset_index(drop=True)

'''
4) add a column BIAS= obs-pm25 - ufs-pm25
'''
df_sort['bias_pm25']= df_sort['airnow_obs_pm25']- df_sort['v13_ufs_pm25']

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

df_sort.to_csv('PM25_2025JanToApr_HourlyData.csv',index=False)





















































