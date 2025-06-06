#!/usr/bin/env python3
"""
Time Series Data Preparation for DAFCOM Models

This module handles the preparation of time series data for training LSTM and Transformer models.
It provides functionality for:
- Creating sequences from time series data
- Handling data normalization and scaling
- Managing batches for efficient processing
- Supporting Dask for distributed data preparation

Author: GitHub Copilot
Date: June 5, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import dask
import dask.array as da
import dask.dataframe as dd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path

# Import utilities from DAFCOM package
from dafcom.utils.validation import validate_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesDataPreparation:
    """Class for preparing time series data for ML models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time series data preparation.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config

        # Extract parameters from config
        self.time_steps = config.get("time_steps", 96)         # Input sequence length
        self.forecast_steps = config.get("forecast_steps", 24) # Output sequence length
        self.total_steps = self.time_steps + self.forecast_steps

        # Feature columns (predictors) and target columns (to predict)
        self.feature_columns = config.get("feature_columns", [])
        self.target_columns = config.get("target_columns", [])

        # Directory for saving scalers
        self.scaler_dir = config.get("scaler_dir", "./scalers")
        os.makedirs(self.scaler_dir, exist_ok=True)

        # Feature scaling configuration
        self.scaling_method = config.get("scaling_method", "standard")  # standard, minmax, robust
        self.feature_scalers = {}
        self.target_scalers = {}

        # Configure whether to use Dask for distributed processing
        self.use_dask = config.get("use_dask", False)
        self.dask_chunk_size = config.get("dask_chunk_size", "100MB")
        self.dask_n_workers = config.get("dask_n_workers", None)

        if self.use_dask:
            dask_scheduler = config.get("dask_scheduler", "processes")
            if dask_scheduler not in ["threads", "processes", "distributed"]:
                logger.warning(f"Invalid dask scheduler: {dask_scheduler}. Using 'processes' instead.")
                dask_scheduler = "processes"
            self.dask_scheduler = dask_scheduler

    def fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Fit scalers to the input dataframe.

        Args:
            df: DataFrame containing both feature and target columns
        """
        logger.info("Fitting data scalers...")

        # Create scalers based on configured method
        scaler_class = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler
        }.get(self.scaling_method, StandardScaler)

        # Fit feature scalers
        for col in self.feature_columns:
            if col in df.columns:
                scaler = scaler_class()
                values = df[col].values.reshape(-1, 1)
                scaler.fit(values)
                self.feature_scalers[col] = scaler
                logger.info(f"Fitted {self.scaling_method} scaler for feature: {col}")

        # Fit target scalers
        for col in self.target_columns:
            if col in df.columns:
                scaler = scaler_class()
                values = df[col].values.reshape(-1, 1)
                scaler.fit(values)
                self.target_scalers[col] = scaler
                logger.info(f"Fitted {self.scaling_method} scaler for target: {col}")

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to the input dataframe.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with scaled values
        """
        transformed_df = df.copy()

        # Transform features
        for col, scaler in self.feature_scalers.items():
            if col in transformed_df.columns:
                values = transformed_df[col].values.reshape(-1, 1)
                transformed_df[col] = scaler.transform(values).flatten()

        # Transform targets
        for col, scaler in self.target_scalers.items():
            if col in transformed_df.columns:
                values = transformed_df[col].values.reshape(-1, 1)
                transformed_df[col] = scaler.transform(values).flatten()

        return transformed_df

    def inverse_transform_targets(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target predictions back to original scale.

        Args:
            y_pred: Array of scaled predictions
                   (shape: [samples, forecast_steps, target_features])

        Returns:
            Array of predictions in original scale
        """
        y_original = np.zeros_like(y_pred)

        # If we have only one target feature, handle differently
        if len(self.target_columns) == 1:
            col = self.target_columns[0]
            if col in self.target_scalers:
                scaler = self.target_scalers[col]

                # Reshape for inverse transform and then back
                samples, steps = y_pred.shape[0], y_pred.shape[1]
                y_reshaped = y_pred.reshape(-1, 1)
                y_original_reshaped = scaler.inverse_transform(y_reshaped)
                y_original = y_original_reshaped.reshape(samples, steps, 1)
        else:
            # For multiple target features
            for i, col in enumerate(self.target_columns):
                if col in self.target_scalers:
                    scaler = self.target_scalers[col]

                    # Extract the ith feature across all samples and times
                    y_feat = y_pred[:, :, i]
                    samples, steps = y_feat.shape
                    y_reshaped = y_feat.reshape(-1, 1)
                    y_original_reshaped = scaler.inverse_transform(y_reshaped)
                    y_original[:, :, i] = y_original_reshaped.reshape(samples, steps)

        return y_original

    def save_scalers(self, path: Optional[str] = None) -> None:
        """
        Save fitted scalers to disk.

        Args:
            path: Directory to save scalers (uses self.scaler_dir if not provided)
        """
        save_dir = path or self.scaler_dir
        os.makedirs(save_dir, exist_ok=True)

        # Save feature scalers
        feature_scalers_path = os.path.join(save_dir, 'feature_scalers.joblib')
        joblib.dump(self.feature_scalers, feature_scalers_path)

        # Save target scalers
        target_scalers_path = os.path.join(save_dir, 'target_scalers.joblib')
        joblib.dump(self.target_scalers, target_scalers_path)

        logger.info(f"Scalers saved to {save_dir}")

    def load_scalers(self, path: Optional[str] = None) -> None:
        """
        Load scalers from disk.

        Args:
            path: Directory containing saved scalers
        """
        load_dir = path or self.scaler_dir

        # Load feature scalers
        feature_scalers_path = os.path.join(load_dir, 'feature_scalers.joblib')
        if os.path.exists(feature_scalers_path):
            self.feature_scalers = joblib.load(feature_scalers_path)
            logger.info(f"Loaded feature scalers from {feature_scalers_path}")
        else:
            logger.warning(f"Feature scalers not found at {feature_scalers_path}")

        # Load target scalers
        target_scalers_path = os.path.join(load_dir, 'target_scalers.joblib')
        if os.path.exists(target_scalers_path):
            self.target_scalers = joblib.load(target_scalers_path)
            logger.info(f"Loaded target scalers from {target_scalers_path}")
        else:
            logger.warning(f"Target scalers not found at {target_scalers_path}")

    def create_sequences(self, data: pd.DataFrame, date_column: str = "date") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for time series forecasting.

        Args:
            data: DataFrame with time series data (should have date_column)
            date_column: Name of the column containing datetime information

        Returns:
            Tuple of (X, y) where X contains input sequences and y contains target sequences
        """
        logger.info("Creating time series sequences...")

        # Validate inputs
        if not all(col in data.columns for col in self.feature_columns):
            missing = [col for col in self.feature_columns if col not in data.columns]
            raise ValueError(f"Missing feature columns in data: {missing}")

        if not all(col in data.columns for col in self.target_columns):
            missing = [col for col in self.target_columns if col not in data.columns]
            raise ValueError(f"Missing target columns in data: {missing}")

        # Extract features and targets
        features = data[self.feature_columns].values
        targets = data[self.target_columns].values

        # Number of valid sequences we can create
        n_samples = len(data) - self.total_steps + 1

        if n_samples <= 0:
            raise ValueError(f"Not enough data points ({len(data)}) to create sequences of length {self.total_steps}")

        # Initialize arrays
        X = np.zeros((n_samples, self.time_steps, len(self.feature_columns)))
        y = np.zeros((n_samples, self.forecast_steps, len(self.target_columns)))

        # Create sequences
        for i in range(n_samples):
            X[i] = features[i:i+self.time_steps]
            y[i] = targets[i+self.time_steps:i+self.total_steps]

        logger.info(f"Created {n_samples} sequences with shape X: {X.shape}, y: {y.shape}")
        return X, y

    def create_sequences_with_dates(self, data: pd.DataFrame, date_column: str = "date") -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Create input-output sequences with date information.

        Args:
            data: DataFrame with time series data (should have date_column)
            date_column: Name of the column containing datetime information

        Returns:
            Tuple of (X, y, sequence_dates) where X contains input sequences,
            y contains target sequences, and sequence_dates has the starting date
            of each sequence
        """
        X, y = self.create_sequences(data, date_column)

        # Extract sequence start dates
        dates = data[date_column].values
        sequence_dates = []

        n_samples = len(X)
        for i in range(n_samples):
            sequence_dates.append(dates[i])

        return X, y, sequence_dates

    def create_dask_sequences(self, data: Union[pd.DataFrame, dd.DataFrame], date_column: str = "date") -> Tuple[da.Array, da.Array]:
        """
        Create input-output sequences using Dask for distributed processing.

        Args:
            data: DataFrame or Dask DataFrame with time series data
            date_column: Name of the column containing datetime information

        Returns:
            Tuple of (X, y) as Dask Arrays
        """
        if not self.use_dask:
            logger.warning("Dask is not enabled in config. Enabling it for this operation.")

        # Convert to dask dataframe if necessary
        if isinstance(data, pd.DataFrame):
            dask_df = dd.from_pandas(data, chunksize=self.dask_chunk_size)
        else:
            dask_df = data

        logger.info("Creating time series sequences with Dask (distributed)...")

        # This is a simplified approach - for very large datasets, you would need a more sophisticated algorithm
        # that properly handles chunk boundaries

        # Create a mapped function to create sequences within each partition
        def create_partition_sequences(partition_df):
            X_list = []
            y_list = []

            # Convert to pandas for easier manipulation
            pdf = partition_df.compute() if hasattr(partition_df, 'compute') else partition_df

            # Sort by date if the column exists
            if date_column in pdf.columns:
                pdf = pdf.sort_values(date_column)

            features = pdf[self.feature_columns].values
            targets = pdf[self.target_columns].values

            # Calculate number of sequences
            n_samples = len(pdf) - self.total_steps + 1

            if n_samples > 0:
                for i in range(n_samples):
                    X_list.append(features[i:i+self.time_steps])
                    y_list.append(targets[i+self.time_steps:i+self.total_steps])

            if X_list:
                return np.array(X_list), np.array(y_list)
            else:
                # Return empty arrays with correct shapes
                return (np.zeros((0, self.time_steps, len(self.feature_columns))),
                        np.zeros((0, self.forecast_steps, len(self.target_columns))))

        # Apply the function to each partition and collect results
        results = []
        for partition in dask_df.partitions:
            results.append(dask.delayed(create_partition_sequences)(partition))

        # Compute all partitions
        computed_results = dask.compute(*results)

        # Combine results
        X_parts = [res[0] for res in computed_results if res[0].shape[0] > 0]
        y_parts = [res[1] for res in computed_results if res[1].shape[0] > 0]

        if not X_parts or not y_parts:
            return da.zeros((0, self.time_steps, len(self.feature_columns))), da.zeros((0, self.forecast_steps, len(self.target_columns)))

        X = da.concatenate([da.from_array(x, chunks=self.dask_chunk_size) for x in X_parts])
        y = da.concatenate([da.from_array(y, chunks=self.dask_chunk_size) for y in y_parts])

        logger.info(f"Created sequences with Dask. X shape: {X.shape}, y shape: {y.shape}")

        return X, y
