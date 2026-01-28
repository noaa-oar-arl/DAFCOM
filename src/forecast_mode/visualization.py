"""
Aero-compliant Visualization Module.
🍃⚡ This module uses Cartopy for geospatial plotting, as mandated by the Aero Protocol.
"""

import os
from datetime import datetime
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.feature as cfeature
except ImportError:
    # We allow import if cartopy is missing but will fail later with clear message
    pass


class Bias_Plotter:
    """
    Class for plotting prediction biases on geospatial maps.

    🍃⚡ Aero Protocol: Uses Cartopy with explicit projections and transforms.
    """

    def __init__(self, csv_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        Initialize the plotter with data from a CSV or DataFrame.

        Parameters
        ----------
        csv_path : str, optional
            Path to the CSV file containing prediction data.
        df : pd.DataFrame, optional
            DataFrame containing prediction data.
        """
        if csv_path is None and df is None:
            raise ValueError("Either csv_path or df must be provided.")
        self.csv_path = csv_path
        self.df = df
        self._data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV or return the provided DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded data.
        """
        if self._data is not None:
            return self._data
        if self.df is not None:
            df = self.df.copy()
        else:
            df = pd.read_csv(self.csv_path)
        # Basic column checks
        required = {"time", "lat", "lon", "xgb_predictions_pm25", "lstm_pm25", "ufs_pm25", "obs_pm25"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        self._data = df
        return df

    def filter_time_range(self, start_time: Union[str, datetime], end_time: Union[str, datetime]) -> pd.DataFrame:
        """
        Filter the data within a specified time range.

        Parameters
        ----------
        start_time : str or datetime
            The start of the time range.
        end_time : str or datetime
            The end of the time range.

        Returns
        -------
        pd.DataFrame
            The filtered data.
        """
        df = self.load_data()
        start = pd.to_datetime(start_time)
        end = pd.to_datetime(end_time)
        df_time = df[(df["time"] >= start) & (df["time"] < end)].copy()
        if df_time.empty:
            raise ValueError("No records found in the specified time window.")
        return df_time

    def compute_biases(self, df_time: pd.DataFrame) -> pd.DataFrame:
        """
        Compute prediction biases for different models.

        Parameters
        ----------
        df_time : pd.DataFrame
            The data to compute biases for.

        Returns
        -------
        pd.DataFrame
            The DataFrame with added bias columns.
        """
        df = df_time.copy()
        df["bias_xgb"] = df["xgb_predictions_pm25"] - df["obs_pm25"]
        df["bias_lstm"] = df["lstm_pm25"] - df["obs_pm25"]
        df["bias_ufs"] = df["ufs_pm25"] - df["obs_pm25"]
        return df

    def plot_bias_maps(
        self,
        df_time: pd.DataFrame,
        vmin: float = -10.0,
        vmax: float = 10.0,
        cmap: str = "coolwarm",
        figsize: Tuple[int, int] = (18, 6),
        dpi: int = 300,
        out_dir: Optional[str] = None,
        out_name: Optional[str] = None,
    ) -> str:
        """
        Create and save a 3-panel bias figure using Cartopy.

        🍃⚡ Aero Protocol Rule 3: Track A (Publication).
        Mandatory: projection= in axes and transform= in plot calls.

        Parameters
        ----------
        df_time : pd.DataFrame
            The data to plot.
        vmin : float, default -10.0
            Minimum value for color scale.
        vmax : float, default 10.0
            Maximum value for color scale.
        cmap : str, default 'coolwarm'
            Colormap to use.
        figsize : tuple, default (18, 6)
            Figure size in inches.
        dpi : int, default 300
            Dots per inch for the saved figure.
        out_dir : str, optional
            Output directory for the plot.
        out_name : str, optional
            Output filename.

        Returns
        -------
        str
            The path to the saved figure.
        """
        try:
            import cartopy.crs as ccrs
        except ImportError as e:
            raise ImportError("Cartopy is required for plotting. Install cartopy.") from e

        lat = np.array(df_time["lat"])
        lon = np.array(df_time["lon"])
        biases = {
            "XGB": np.array(df_time["bias_xgb"]),
            "LSTM": np.array(df_time["bias_lstm"]),
            "UFS-AQM": np.array(df_time["bias_ufs"]),
        }

        # Projection for the map
        map_proj = ccrs.Mercator()
        # Data projection (usually PlateCarree for lat/lon)
        data_proj = ccrs.PlateCarree()

        fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw={"projection": map_proj})

        file_time_str = str(pd.to_datetime(df_time["time"].iloc[0])).replace(" 00:00:00", "")

        for ax, (model_name, values) in zip(axs, biases.items()):
            ax.set_extent([-125, -65, 24, 49], crs=data_proj)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=":")

            sc = ax.scatter(
                lon,
                lat,
                c=values,
                cmap=cmap,
                s=30,
                edgecolors="k",
                vmin=vmin,
                vmax=vmax,
                transform=data_proj,
            )

            cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
            cbar.set_label("Predict Bias")
            ax.set_title(f"PM2.5 Bias ({model_name} - Obs)\n{file_time_str}")

        plt.tight_layout()

        if out_dir is None:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        if out_name is None:
            out_name = f"{file_time_str}_predict_bias_pm25.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path
