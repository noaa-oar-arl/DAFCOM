"""
Aero-compliant Evaluator Module.
🍃⚡ This module provides evaluation metrics for air quality forecasts.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error


class Evaluator:
    """
    Evaluator class for comparing model predictions against observations.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        obs_col: str = "obs_pm25",
        model_cols: Optional[List[str]] = None,
    ):
        """
        Initialize the evaluator with data and column mappings.

        Parameters
        ----------
        csv_path : str, optional
            Path to the CSV file containing prediction data.
        df : pd.DataFrame, optional
            DataFrame containing prediction data.
        obs_col : str, default 'obs_pm25'
            Name of the column containing observations.
        model_cols : List[str], optional
            Names of the columns containing model predictions.
        """
        self.csv_path = csv_path
        self.df = df
        self.obs_col = obs_col
        self.model_cols = model_cols or ["xgb_predictions_pm25", "lstm_pm25", "ufs_pm25"]
        self._data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Dict[str, float]] = {}

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
            self._data = self.df.copy()
        elif self.csv_path:
            self._data = pd.read_csv(self.csv_path)
        else:
            raise ValueError("Either csv_path or df must be provided.")
        # validate presence of columns
        missing = [c for c in [self.obs_col] + self.model_cols if c not in self._data.columns]
        if missing:
            raise KeyError(f"Missing columns in input data: {missing}")
        return self._data

    @staticmethod
    def compute_pair_metrics(obs: Union[np.ndarray, pd.Series], pred: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Compute metrics for a pair of 1D arrays (obs, pred).

        Parameters
        ----------
        obs : np.ndarray or pd.Series
            Observed values.
        pred : np.ndarray or pd.Series
            Predicted values.

        Returns
        -------
        Dict[str, Any]
            Dictionary of computed metrics (r, mse, mean_bias, std_model, std_obs, n).
        """
        # drop NaNs in either
        obs = np.asarray(obs, dtype=float)
        pred = np.asarray(pred, dtype=float)
        mask = ~np.isnan(obs) & ~np.isnan(pred)
        if mask.sum() < 2:
            nan = float("nan")
            return {"r": nan, "mse": nan, "mean_bias": nan, "std_model": nan, "std_obs": nan, "n": int(mask.sum())}
        obs_clean = obs[mask]
        pred_clean = pred[mask]
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(obs_clean, pred_clean)
            r = float(r_value)
        except Exception:
            r = float("nan")
        mse = float(mean_squared_error(obs_clean, pred_clean))
        mean_bias = float(np.mean(pred_clean - obs_clean))
        std_model = float(np.std(pred_clean))
        std_obs = float(np.std(obs_clean))
        return {"r": r, "mse": mse, "mean_bias": mean_bias, "std_model": std_model, "std_obs": std_obs, "n": int(mask.sum())}

    def evaluate_all(self, csv_path: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load data (if needed) and evaluate all model columns against obs_col.

        Parameters
        ----------
        csv_path : str, optional
            Path to the CSV file (overrides initialization).
        df : pd.DataFrame, optional
            DataFrame (overrides initialization).

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Metrics for each model column.
        """
        if df is not None:
            self.df = df
            self._data = None
        if csv_path is not None:
            self.csv_path = csv_path
            self._data = None
        data = self.load_data()
        obs = data[self.obs_col].values
        results = {}
        for col in self.model_cols:
            pred = data[col].values
            metrics = self.compute_pair_metrics(obs, pred)
            results[col] = metrics
        self.results = results
        return results
