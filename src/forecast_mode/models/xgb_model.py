"""
Aero-compliant XGBoost Pipeline Module.
🍃⚡ This module provides a pipeline for training and tuning XGBoost models for air quality forecasting.
"""

import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

######################
# define constants
######################

DEFAULT_FEATURES = ["aod", "pressfc", "tmp2m", "spfh2m", "hpbl", "ugrd10m", "vgrd10m"]
DEFAULT_TARGET = "pm25"
DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


class XGBPipeline:
    """
    XGBoost Training Pipeline.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        df: Optional[Union[pd.DataFrame, xr.Dataset, xr.DataArray]] = None,
        time_step: int = 168,
        time_step_short: int = 96,
        features: Optional[List[str]] = None,
        target: str = DEFAULT_TARGET,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        model_path: str = "xgb_model.json",
        random_state: Optional[int] = 42,
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        data_path : str, optional
            Path to the CSV data file.
        df : pd.DataFrame or xr.Dataset or xr.DataArray, optional
            Data containing training samples.
        time_step : int, default 168
            Total time steps (Input + Forecast). 168 = 96h input + 72h forecast.
        time_step_short : int, default 96
            Input sequence length (4 days = 96h).
        features : List[str], optional
            List of feature names.
        target : str, default 'pm25'
            Target variable name.
        param_grid : Dict[str, List[Any]], optional
            Hyperparameter grid for search.
        model_path : str, default 'xgb_model.json'
            Path to save/load the model.
        random_state : int, default 42
            Random seed for reproducibility.
        """
        self.data_path = data_path
        self.df = df
        self.time_step = time_step
        self.time_step_short = time_step_short
        self.features = features or DEFAULT_FEATURES
        self.target = target
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.model_path = model_path
        self.random_state = random_state

        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.model: Optional[xgb.Booster] = None
        self.searcher: Optional[RandomizedSearchCV] = None

    def _to_dataframe(self, data: Union[pd.DataFrame, xr.Dataset, xr.DataArray]) -> pd.DataFrame:
        """
        Convert xarray objects to pandas DataFrame if necessary.

        Parameters
        ----------
        data : pd.DataFrame or xr.Dataset or xr.DataArray
            Input data.

        Returns
        -------
        pd.DataFrame
            Converted DataFrame.
        """
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            # 🍃⚡ Aero Protocol: Convert to dataframe only for ML backend.
            return data.to_dataframe().reset_index()
        return data.copy()

    def prepare_sequences(
        self, df: Optional[Union[pd.DataFrame, xr.Dataset, xr.DataArray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for XGBoost training by flattening time dimension.

        Parameters
        ----------
        df : pd.DataFrame or xr.Dataset or xr.DataArray, optional
            The data to prepare. If None, uses loaded data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Features matrix (X) and target vector (Y).
        """
        if df is None:
            df = self.df if self.df is not None else self.data_path

        if isinstance(df, str):
            df = pd.read_csv(df)
        elif isinstance(df, (xr.Dataset, xr.DataArray)):
            # 🍃⚡ Aero Protocol: Convert to dataframe only when needed for non-Xarray backend.
            df = df.to_dataframe().reset_index()
        elif isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            raise ValueError("No data provided or unsupported format")

        df["time_utc"] = pd.to_datetime(df["time_utc"], format="%Y-%m-%d %H:%M:%S")

        site_list = list(df["site_index"])
        time_list = list(df["time_utc"])

        sequence_line_start_list = []
        for i in range(len(site_list)):
            if i + self.time_step <= len(site_list):
                if site_list[i + self.time_step - 1] == site_list[i]:
                    time_start = time_list[i]
                    time_end = time_list[i + self.time_step - 1]
                    if time_start + timedelta(hours=self.time_step - 1) == time_end:
                        sequence_line_start_list.append(i)

        data_x_raw = df[self.features].values
        data_y_raw = df[self.target].values

        X = []
        Y = []
        for start in sequence_line_start_list:
            # Flatten X: (time_step_short, n_features) -> (time_step_short * n_features)
            X.append(data_x_raw[start : start + self.time_step_short, :].flatten())
            # Y: (forecast_length,)
            Y.append(data_y_raw[start + self.time_step_short : start + self.time_step])

        self.X = np.array(X)
        self.Y = np.array(Y)

        return self.X, self.Y

    def load_existing_model(self) -> Optional[xgb.Booster]:
        """
        Load an existing model from self.model_path if it exists.

        Returns
        -------
        xgb.Booster or None
            The loaded model, or None if not found.
        """
        if os.path.exists(self.model_path):
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            return self.model
        return None

    def train_model(self, params: Optional[Dict] = None, num_boost_round: int = 10) -> xgb.Booster:
        """
        Train the XGBoost model. Supports continual learning.

        Parameters
        ----------
        params : Dict, optional
            Training parameters.
        num_boost_round : int, default 10
            Number of boosting rounds.

        Returns
        -------
        xgb.Booster
            The trained model.
        """
        if self.X is None or self.Y is None:
            raise RuntimeError("Call prepare_sequences first")

        dtrain = xgb.DMatrix(self.X, label=self.Y)
        existing_model = self.load_existing_model()

        default_params = {
            "objective": "reg:squarederror",
            "seed": self.random_state,
            "learning_rate": 0.1,
            "max_depth": 6,
        }
        if params:
            default_params.update(params)

        self.model = xgb.train(default_params, dtrain, num_boost_round=num_boost_round, xgb_model=existing_model)
        self.model.save_model(self.model_path)
        return self.model

    def train_iterative(self, df: Union[pd.DataFrame, xr.Dataset, xr.DataArray], num_boost_round: int = 5) -> xgb.Booster:
        """
        Perform one iteration of training on a new batch of data.

        Parameters
        ----------
        df : pd.DataFrame or xr.Dataset or xr.DataArray
            The new batch of data.
        num_boost_round : int, default 5
            Number of boosting rounds for this iteration.

        Returns
        -------
        xgb.Booster
            The trained model.
        """
        X, Y = self.prepare_sequences(df)
        self.X, self.Y = X, Y
        return self.train_model(num_boost_round=num_boost_round)

    def tune_random_search(
        self,
        n_iter: int = 300,
        cv: int = 10,
        scoring: str = "r2",
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> RandomizedSearchCV:
        """
        Run RandomizedSearchCV on XGBRegressor and store the searcher.

        Parameters
        ----------
        n_iter : int, default 300
            Number of parameter settings that are sampled.
        cv : int, default 10
            Number of cross-validation folds.
        scoring : str, default 'r2'
            Scoring metric.
        n_jobs : int, default -1
            Number of jobs to run in parallel.
        verbose : int, default 1
            Verbosity level.

        Returns
        -------
        RandomizedSearchCV
            The fitted search object.
        """
        if self.X is None or self.Y is None:
            raise RuntimeError("Call prepare_sequences() first.")
        xgb_base = xgb.XGBRegressor(objective="reg:squarederror", random_state=self.random_state)
        search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=self.param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=self.random_state,
        )
        search.fit(self.X, self.Y)
        self.searcher = search
        return search

    def fit_best_model(self) -> xgb.XGBRegressor:
        """
        Instantiate estimator with best params and fit on all data.

        Returns
        -------
        xgb.XGBRegressor
            The fitted best model.
        """
        if self.searcher is None:
            raise RuntimeError("Call tune_random_search() first.")
        best_params = self.searcher.best_params_
        best = xgb.XGBRegressor(objective="reg:squarederror", random_state=self.random_state, **best_params)
        best.fit(self.X, self.Y)
        return best

    def save_model(self, out_path: str = "./xgb_model.joblib") -> str:
        """
        Save the trained Booster to disk.

        Parameters
        ----------
        out_path : str, default './xgb_model.joblib'
            Output path for the saved model.

        Returns
        -------
        str
            The path to the saved model.
        """
        if self.model is None:
            raise RuntimeError("No trained model to save. Call train_model() or train_iterative() first.")
        self.model.save_model(out_path)
        return out_path

    def run(
        self,
        n_iter: int = 300,
        cv: int = 10,
        scoring: str = "r2",
        n_jobs: int = -1,
        verbose: int = 1,
        model_out: str = "./xgb_model.json",
    ) -> Dict[str, Any]:
        """
        Orchestrator: prepare data, tune, fit best, save model.

        Parameters
        ----------
        n_iter : int, default 300
            Number of parameter settings for random search.
        cv : int, default 10
            Cross-validation folds.
        scoring : str, default 'r2'
            Scoring metric.
        n_jobs : int, default -1
            Parallel jobs.
        verbose : int, default 1
            Verbosity.
        model_out : str, default './xgb_model.json'
            Output path for model.

        Returns
        -------
        Dict[str, Any]
            Results dictionary (best_params, best_score, model_path).
        """
        self.prepare_sequences()
        search = self.tune_random_search(n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        best_score = search.best_score_
        best_params = search.best_params_
        best_estimator = self.fit_best_model()
        self.model = best_estimator.get_booster()
        model_path = self.save_model(model_out)
        return {
            "best_params": best_params,
            "best_score": best_score,
            "model_path": model_path,
        }
