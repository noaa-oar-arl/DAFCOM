# -*- coding: utf-8 -*-
"""
Created on Fri May 30 11:29:59 2025

@author: Beiming.Tang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime as dt
from datetime import timedelta

dir_source = '/data/aqf3/beiming.tang/DAFCOM/code/step8_compare_to_BiasCorrection/site_only/'
pattern_csv = '2_merged_xbg_and_pmbc.csv'

df = pd.read_csv(dir_source+pattern_csv)


day_list = [7,8,9,10,
            11,12,13,14,15,16,17,18,19,20,
            21,22,23,24,25,26,27,28,29,30,
            31]
for day in day_list:
    start_time = dt(2025,5,day,0,0,0)
    #end_time = dt(2025,5,8,0,0,0)
    end_time = start_time + timedelta(days=1)

    file_name = str(start_time).replace(' 00:00:00','')
    df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%d %H:%M:%S')
    df_time =  df[(df['time']>=start_time) & (df['time'] < end_time)]


    '''
    variable
    '''
    xgb_pm25 = np.array(df_time['xgb_predictions_pm25'])
    bc_pm25 = np.array(df_time['bias_correction_pm25'])
    lat = np.array(df_time['lat_x'])
    lon = np.array(df_time['lon_x'])
    obs_pm25 = np.array(df_time['obs_pm25'])
    ufs_pm25 = np.array(df_time['ufs_pm25'])
    lstm_pm25 = np.array(df_time['lstm_pm25'])



    bias_xgb_pm25 = xgb_pm25 - obs_pm25
    bias_bc_pm25 = bc_pm25 - obs_pm25
    bias_lstm_pm25 = lstm_pm25 - obs_pm25
    bias_ufs_pm25 = ufs_pm25 - obs_pm25


    '''
    plot
    '''

    fig, axs = plt.subplots(1,3, figsize = (15,6))

    '''
    1st plot
    '''
    m1 = Basemap(ax=axs[0],projection = 'merc',llcrnrlat = 24, urcrnrlat = 49, llcrnrlon = -125, urcrnrlon = -65, resolution = 'l')
    m1.drawcoastlines()
    m1.drawstates()
    m1.drawcounties()
    m1.drawcountries()

    x,y = m1(lon, lat)
    sc = m1.scatter(x,y, c= bias_xgb_pm25, cmap='coolwarm',s=30,edgecolors='k',vmin = -10, vmax = 10)
    plt.colorbar(sc, label = 'predict bias',orientation='horizontal')

    axs[0].set_title ('PM2.5 bias (XGB - obs)\n'+file_name)

    '''
    2nd plot
    '''
    m2 = Basemap(ax=axs[1],projection = 'merc',llcrnrlat = 24, urcrnrlat = 49, llcrnrlon = -125, urcrnrlon = -65, resolution = 'l')
    m2.drawcoastlines()
    m2.drawstates()
    m2.drawcounties()
    m2.drawcountries()

    x,y = m2(lon, lat)
    sc = m2.scatter(x,y, c= bias_bc_pm25, cmap='coolwarm',s=30,edgecolors='k',vmin = -10, vmax = 10)
    plt.colorbar(sc, label = 'predict bias',orientation='horizontal')

    axs[1].set_title ('PM2.5 bias (bias correction - obs)\n'+file_name)

    '''
    3rd plot
    '''
    m3 = Basemap(ax=axs[2],projection = 'merc',llcrnrlat = 24, urcrnrlat = 49, llcrnrlon = -125, urcrnrlon = -65, resolution = 'l')
    m3.drawcoastlines()
    m3.drawstates()
    m3.drawcounties()
    m3.drawcountries()

    x,y = m3(lon, lat)
    sc = m3.scatter(x,y, c= bias_ufs_pm25, cmap='coolwarm',s=30,edgecolors='k',vmin = -10, vmax = 10)
    plt.colorbar(sc, label = 'predict bias',orientation='horizontal')

    axs[2].set_title ('PM2.5 bias (UFS-AQM - obs)\n'+file_name)

    plt.tight_layout()

    dir_save = dir_source+'plots/'
    plt.savefig(dir_save+file_name+'_predict_bias_pm25.png',dpi = 600)














