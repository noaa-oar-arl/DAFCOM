#################
# Import Library
#################

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

############################
# Define Class and Functions
############################
class Bias_Plotter:

    def __init__(self, csv_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        if csv_path is None and df is None:
            raise ValueError("Either csv_path or df must be provided.")
        self.csv_path = csv_path
        self.df = df
        self._data = None

    def load_data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        if self.df is not None:
            df = self.df.copy()
        else:
            df = pd.read_csv(self.csv_path)
        # Basic column checks
        required = {'time', 'lat', 'lon', 'xgb_predictions_pm25', 'lstm_pm25', 'ufs_pm25', 'obs_pm25'}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        self._data = df
        return df

    def filter_time_range(self, start_time: str | datetime, end_time: str | datetime) -> pd.DataFrame:
        df = self.load_data()
        start = pd.to_datetime(start_time)
        end = pd.to_datetime(end_time)
        df_time = df[(df['time'] >= start) & (df['time'] < end)].copy()
        if df_time.empty:
            raise ValueError("No records found in the specified time window.")
        return df_time

    def compute_biases(self, df_time: pd.DataFrame) -> pd.DataFrame:
        df = df_time.copy()
        df['bias_xgb'] = df['xgb_predictions_pm25'] - df['obs_pm25']
        df['bias_lstm'] = df['lstm_pm25'] - df['obs_pm25']
        df['bias_ufs'] = df['ufs_pm25'] - df['obs_pm25']
        return df

    def plot_bias_maps(
        self,
        df_time: pd.DataFrame,
        vmin: float = -10.0,
        vmax: float = 10.0,
        cmap: str = 'coolwarm',
        figsize: Tuple[int, int] = (15, 6),
        dpi: int = 600,
        out_dir: Optional[str] = None,
        out_name: Optional[str] = None,
        basemap_kwargs: Optional[dict] = None,
    ) -> str:
        """
        Create and save the 3-panel bias figure. Returns path to saved file.
        """
        # import Basemap lazily because some environments don't have it
        try:
            from mpl_toolkits.basemap import Basemap
        except Exception as e:
            raise ImportError("Basemap is required for plotting. Install mpl_toolkits.basemap.") from e

        lat = np.array(df_time['lat'])
        lon = np.array(df_time['lon'])
        bias_xgb = np.array(df_time['bias_xgb'])
        bias_lstm = np.array(df_time['bias_lstm'])
        bias_ufs = np.array(df_time['bias_ufs'])

        bm_kw = dict(llcrnrlat=24, urcrnrlat=49, llcrnrlon=-125, urcrnrlon=-65, resolution='l')
        if basemap_kwargs:
            bm_kw.update(basemap_kwargs)

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # helper to draw a map and scatter
        def _draw_map(ax, values, title):
            m = Basemap(ax=ax, projection='merc', **bm_kw)
            m.drawcoastlines()
            m.drawstates()
            m.drawcounties()
            m.drawcountries()
            x, y = m(lon, lat)
            sc = m.scatter(x, y, c=values, cmap=cmap, s=30, edgecolors='k', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label('predict bias')
            ax.set_title(title)

        file_time_str = str(pd.to_datetime(df_time['time'].iloc[0])).replace(' 00:00:00', '')
        _draw_map(axs[0], bias_xgb, f'PM2.5 bias (XGB - obs)\n{file_time_str}')
        _draw_map(axs[1], bias_lstm, f'PM2.5 bias (LSTM - obs)\n{file_time_str}')
        _draw_map(axs[2], bias_ufs, f'PM2.5 bias (UFS-AQM - obs)\n{file_time_str}')

        plt.tight_layout()

        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        if out_name is None:
            out_name = f"{file_time_str}_predict_bias_pm25.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return out_path


