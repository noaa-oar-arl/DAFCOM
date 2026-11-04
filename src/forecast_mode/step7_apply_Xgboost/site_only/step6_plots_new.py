
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime as dt, timedelta
from pathlib import Path

dir_source = '/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Data/step7_xgb_apply/site_only/'
pattern_csv = 'xgb_lstm_ufs_predictions_May04to31.csv'

dir_output = '/scratch3/NCEPDEV/stmp/Beiming.Tang/Project/Dafcom/Data/step7_xgb_apply/site_only/plot/'
Path(dir_output).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(dir_source + pattern_csv)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

# Loop from May 4 to May 31 (inclusive)
for start_time in pd.date_range('2026-05-05', '2026-05-31', freq='D'):

    end_time = start_time + timedelta(days=1)
    file_name = start_time.strftime('%Y-%m-%d')

    print(f'Processing {file_name}', flush=True)

    df_time = df[
        (df['time'] >= start_time) &
        (df['time'] < end_time)
    ]

    if df_time.empty:
        print(f'No data for {file_name}, skipping.')
        continue

    '''
    Variables
    '''
    xgb_pm25 = np.array(df_time['xgb_predictions_pm25'])
    lat = np.array(df_time['lat'])
    lon = np.array(df_time['lon'])
    obs_pm25 = np.array(df_time['obs_pm25'])
    ufs_pm25 = np.array(df_time['ufs_pm25'])
    lstm_pm25 = np.array(df_time['lstm_pm25'])

    bias_xgb_pm25 = xgb_pm25 - obs_pm25
    bias_lstm_pm25 = lstm_pm25 - obs_pm25
    bias_ufs_pm25 = ufs_pm25 - obs_pm25

    '''
    Plot
    '''
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    plot_data = [
        (bias_xgb_pm25, 'XGB - obs'),
        (bias_lstm_pm25, 'LSTM - obs'),
        (bias_ufs_pm25, 'UFS-AQM - obs')
    ]

    for ax, (bias, title) in zip(axs, plot_data):

        m = Basemap(
            ax=ax,
            projection='merc',
            llcrnrlat=24,
            urcrnrlat=49,
            llcrnrlon=-125,
            urcrnrlon=-65,
            resolution='l'
        )

        m.drawcoastlines()
        m.drawstates()
        m.drawcounties()
        m.drawcountries()

        x, y = m(lon, lat)

        sc = m.scatter(
            x, y,
            c=bias,
            cmap='coolwarm',
            s=30,
            edgecolors='k',
            vmin=-5,
            vmax=5
        )

        fig.colorbar(
            sc,
            ax=ax,
            label='predict bias',
            orientation='horizontal'
        )

        ax.set_title(f'PM2.5 bias ({title})\n{file_name}')

    plt.tight_layout()

    output_file = dir_output + file_name + '_predict_bias_pm25.png'

    fig.savefig(output_file, dpi=600, bbox_inches='tight')

    # Important for batch plotting on URSA
    plt.close(fig)

    print(f'Saved: {output_file}', flush=True)

print('All daily plots completed!')

