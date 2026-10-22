# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:34:56 2025

@author: Beiming.Tang
"""

import numpy as np
import pandas as pd

dir_ = '/data/aqf3/beiming.tang/DAFCOM/code/step7_apply_Xgboost/site_only/'
pattern1 = 'PM25_2025_May01to07_HourlyData_LOG.csv'
df1 = pd.read_csv(dir_+pattern1)

pattern2 = 'lstm_predict_siteonly_logPM.csv'
df2 = pd.read_csv(dir_+pattern2)

merge_df = pd.merge(df1, df2, on=['site_index','time_utc'],how='inner')   #only keep matching rows


# test_df = merge_df.dropna()
merge_df.to_csv('3_merged_file_for_xgboost.csv',index=False)
