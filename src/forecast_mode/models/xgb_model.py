"""
Aero-compliant XGBoost Pipeline Module.
🍃⚡ This module provides a pipeline for training and tuning XGBoost models for air quality forecasting.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
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
        df: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        target: str = DEFAULT_TARGET,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        random_state: Optional[int] = 42,
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        data_path : str, optional
            Path to the CSV data file.
        df : pd.DataFrame, optional
            DataFrame containing training data.
        features : List[str], optional
            List of feature names.
        target : str, default 'pm25'
            Target variable name.
        param_grid : Dict[str, List[Any]], optional
            Hyperparameter grid for search.
        random_state : int, default 42
            Random seed for reproducibility.
        """
        self.data_path = data_path
        self.df = df
        self.features = features or DEFAULT_FEATURES
        self.target = target
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.random_state = random_state

        self.input_X: Optional[np.ndarray] = None
        self.output_y: Optional[np.ndarray] = None
        self.searcher: Optional[RandomizedSearchCV] = None
        self.best_estimator_: Optional[xgb.XGBRegressor] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV from path or use provided DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded training data.
        """
        if self.df is not None:
            return self.df.copy()
        if not self.data_path:
            raise ValueError("Either data_path or df must be provided.")
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return pd.read_csv(self.data_path)

    def prepare_inputs(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input matrix X and target vector y.

        Parameters
        ----------
        df : pd.DataFrame, optional
            The data to prepare. If None, uses loaded data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Features matrix (X) and target vector (y).
        """
        df = df if df is not None else self.load_data()
        missing = [c for c in self.features + [self.target] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in data: {missing}")
        X = df[self.features].values.astype(float)
        y = df[self.target].values.astype(float)
        self.input_X = X
        self.output_y = y
        return X, y

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
        if self.input_X is None or self.output_y is None:
            raise RuntimeError("Call prepare_inputs() first.")
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
        search.fit(self.input_X, self.output_y)
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
        best.fit(self.input_X, self.output_y)
        self.best_estimator_ = best
        return best

    def save_model(self, out_path: str = "./xgb_model.joblib") -> str:
        """
        Save the trained model to disk.

        Parameters
        ----------
        out_path : str, default './xgb_model.joblib'
            Output path for the saved model.

        Returns
        -------
        str
            The path to the saved model.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("No fitted model to save. Call fit_best_model() first.")
        joblib.dump(self.best_estimator_, out_path)
        return out_path

    def run(
        self,
        n_iter: int = 300,
        cv: int = 10,
        scoring: str = "r2",
        n_jobs: int = -1,
        verbose: int = 1,
        model_out: str = "./xgb_model.joblib",
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
        model_out : str, default './xgb_model.joblib'
            Output path for model.

        Returns
        -------
        Dict[str, Any]
            Results dictionary (best_params, best_score, model_path).
        """
        self.prepare_inputs()
        search = self.tune_random_search(n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        best_score = search.best_score_
        best_params = search.best_params_
        self.fit_best_model()
        model_path = self.save_model(model_out)
        return {
            "best_params": best_params,
            "best_score": best_score,
            "model_path": model_path,
        }
