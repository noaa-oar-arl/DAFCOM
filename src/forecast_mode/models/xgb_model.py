###################
# import library
###################

from typing import List, Optional, Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import joblib

######################
# define function
######################

class XGBPipeline:

    def __init__(
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        target: str = DEFAULT_TARGET,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        random_state: Optional[int] = 42,
    ):
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
        """Load CSV from path or use provided DataFrame."""
        if self.df is not None:
            return self.df.copy()
        if not self.data_path:
            raise ValueError("Either data_path or df must be provided.")
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return pd.read_csv(self.data_path)

    def prepare_inputs(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input matrix X and target vector y using `self.features` and `self.target`."""
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
        scoring: str = 'r2',
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> RandomizedSearchCV:
        """Run RandomizedSearchCV on XGBRegressor and store the searcher."""
        if self.input_X is None or self.output_y is None:
            raise RuntimeError("Call prepare_inputs() first.")
        xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
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
        """Instantiate estimator with best params and fit on all data."""
        if self.searcher is None:
            raise RuntimeError("Call tune_random_search() first.")
        best_params = self.searcher.best_params_
        best = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state, **best_params)
        best.fit(self.input_X, self.output_y)
        self.best_estimator_ = best
        return best

    def save_model(self, out_path: str = './xgb_model.joblib') -> str:
        """Save the trained model to disk and return the path."""
        if self.best_estimator_ is None:
            raise RuntimeError("No fitted model to save. Call fit_best_model() first.")
        joblib.dump(self.best_estimator_, out_path)
        return out_path

    def run(
        self,
        n_iter: int = 300,
        cv: int = 10,
        scoring: str = 'r2',
        n_jobs: int = -1,
        verbose: int = 1,
        model_out: str = './xgb_model.joblib',
    ) -> Dict[str, Any]:
        """
        Orchestrator: prepare data, tune, fit best, save model.

        Returns:
            dict with keys: best_params, best_score, model_path
        """
        self.prepare_inputs()
        search = self.tune_random_search(n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        best_score = search.best_score_
        best_params = search.best_params_
        self.fit_best_model()
        model_path = self.save_model(model_out)
        return {
            'best_params': best_params,
            'best_score': best_score,
            'model_path': model_path,
        }


