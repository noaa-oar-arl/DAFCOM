# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:34:56 2025

@author: Beiming.Tang
"""

import numpy as np
import pandas as pd

dir_xgb = '/data/aqf3/beiming.tang/DAFCOM/code/step7_apply_Xgboost/site_only/0501_to_0531/'
pattern1 = '4_xgb_lstm_ufs_predictions.csv'
df1 = pd.read_csv(dir_xgb+pattern1)
print('df1',len(df1))

dir_pmbc = '/data/aqf3/beiming.tang/DAFCOM/code/step8_compare_to_BiasCorrection/site_only/'
pattern2 = '1_2025_May_01to31_HourlyData_PM25_BC.csv'
df2 = pd.read_csv(dir_pmbc+pattern2)
print('df2',len(df2))


merge_df = pd.merge(df1, df2, on=['site_index','time'],how='inner')   #only keep matching rows
print('merge',len(merge_df))

# test_df = merge_df.dropna()
merge_df.to_csv('2_merged_xbg_and_pmbc.csv',index=False)
