# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:34:56 2025

@author: Beiming.Tang
"""

import numpy as np
import pandas as pd

dir_ = '/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Data/step7_xgb_apply/site_only/'
pattern1 = 'PM25_2026_May01to31_HourlyData_LOG.csv'
df1 = pd.read_csv(dir_+pattern1)

pattern2 = 'lstm_predict_siteonly_May04to31_logPM.csv'
df2 = pd.read_csv(dir_+pattern2)

merge_df = pd.merge(df1, df2, on=['site_index','time_utc'],how='inner')   #only keep matching rows


# test_df = merge_df.dropna()
merge_df.to_csv(dir_+'3_merged_file_for_xgboost_May04to31.csv',index=False)
