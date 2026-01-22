#===================
# Import Library
#===================
from typing import List, Optional
import numpy as np
import pandas as pd
import os

#===========================
# Define Class and Functions
#===========================
class PM25Preprocessor:
    """
    Load, merge, clean and prepare PM2.5 Excel files.
    """

    def __init__(
        self,
        dir_: str,
        day_list: Optional[List[str]] = None,
        pattern_prefix: str = 'CONUS_PM25_2025_',
        output_csv: str = 'PM25_2025JanToApr_HourlyData_LOG.csv',
        dropna: bool = True
    ):
        self.dir_ = dir_
        self.day_list = day_list if day_list is not None else list(self.DEFAULT_DAY_LIST)
        self.pattern_prefix = pattern_prefix
        self.output_csv = output_csv
        self.dropna = dropna
        self.df_total: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None

    def load_and_merge(self) -> pd.DataFrame:
        """Read each day's Excel and concat into a single DataFrame."""
        frames = []
        for day in self.day_list:
            fname = os.path.join(self.dir_, f"{self.pattern_prefix}{day}.xlsx")
            if not os.path.isfile(fname):
                # skip missing files but warn
                print(f"Warning: file not found, skipping: {fname}")
                continue
            try:
                df_day = pd.read_excel(fname)
            except Exception as e:
                print(f"Warning: failed to read {fname}: {e}")
                continue
            frames.append(df_day)
        if not frames:
            raise FileNotFoundError("No input files were found or readable in the provided directory/day_list.")
        self.df_total = pd.concat(frames, ignore_index=True)
        return self.df_total

    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Drop NaNs, remove 'no_value' for v1_blh and drop duplicates."""
        if df is None:
            if self.df_total is None:
                raise RuntimeError("Call load_and_merge() first or provide a DataFrame")
            df = self.df_total.copy()
        if self.dropna:
            df = df.dropna()
        # keep rows where v1_blh != 'no_value' if column exists
        if 'v1_blh' in df.columns:
            df = df[df['v1_blh'] != 'no_value']
        # drop duplicates on ('site_index','time_utc')
        if {'site_index', 'time_utc'}.issubset(df.columns):
            df = df.drop_duplicates(subset=['site_index', 'time_utc'], keep='first')
        else:
            print("Warning: required columns 'site_index' and/or 'time_utc' not found for duplicate removal.")
        self.df_processed = df
        return df

    def sort(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Sort by site_index and time_utc and reset index."""
        if df is None:
            if self.df_processed is None:
                raise RuntimeError("Call clean() first or provide a DataFrame")
            df = self.df_processed
        if {'site_index', 'time_utc'}.issubset(df.columns):
            df = df.sort_values(by=['site_index', 'time_utc']).reset_index(drop=True)
        else:
            print("Warning: cannot sort because 'site_index' or 'time_utc' missing.")
        self.df_processed = df
        return df

    def add_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add features:
         - log_pm25 = log(airnow_obs_pm25)
         - hour_utc, day_of_year from time_utc
        Keeps safe checks for column existence.
        """
        if df is None:
            if self.df_processed is None:
                raise RuntimeError("Call sort() first or provide a DataFrame")
            df = self.df_processed

        # compute log_pm25
        if 'airnow_obs_pm25' in df.columns:
            # guard against non-positive values
            safe_vals = df['airnow_obs_pm25'].astype(float)
            safe_vals = safe_vals.where(safe_vals > 0, np.nan)
            df['log_pm25'] = np.log(safe_vals)
        else:
            print("Warning: 'airnow_obs_pm25' column not found; 'log_pm25' not created.")

        # parse time and create hour and day_of_year
        if 'time_utc' in df.columns:
            time_utc = pd.to_datetime(df['time_utc'], errors='coerce')
            df['hour_utc'] = time_utc.dt.hour
            df['day_of_year'] = time_utc.dt.dayofyear
        else:
            print("Warning: 'time_utc' column not found; 'hour_utc'/'day_of_year' not created.")

        self.df_processed = df
        return df

    def save_csv(self, df: Optional[pd.DataFrame] = None, output_path: Optional[str] = None) -> str:
        """Save processed dataframe to CSV (default: instance.output_csv). Returns path."""
        if df is None:
            if self.df_processed is None:
                raise RuntimeError("No processed DataFrame to save")
            df = self.df_processed
        out = output_path if output_path is not None else self.output_csv
        df.to_csv(out, index=False)
        return out

    def process(self, save: bool = True, output_path: Optional[str] = None) -> pd.DataFrame:
        """Full pipeline: load, clean, sort, add features, optionally save CSV."""
        self.load_and_merge()
        self.clean()
        self.sort()
        self.add_features()
        if save:
            self.save_csv(output_path)
        return self.df_processed






















