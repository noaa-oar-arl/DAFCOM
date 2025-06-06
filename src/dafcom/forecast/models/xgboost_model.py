#!/usr/bin/env python3
"""
DAFCOM XGBoost Models Module

This module provides functionality for training and applying XGBoost models
for air quality forecasting, often used as a second stage after LSTM models
for further bias correction.

Author: GitHub Copilot based on original by Beiming Tang
Date: June 6, 2025
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import logging
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost model for air quality forecasting and bias correction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the XGBoost model.

        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config
        self.model = None
        self.feature_columns = config.get('feature_columns', [])
        self.target_column = config.get('target_column', 'log_pm25')
        self.model_path = config.get('model_path', './models')

        # XGBoost parameters
        self.params = {
            'objective': config.get('objective', 'reg:squarederror'),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'n_estimators': config.get('n_estimators', 100),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'gamma': config.get('gamma', 0),
            'min_child_weight': config.get('min_child_weight', 1),
            'seed': config.get('seed', 42)
        }

        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

    def train(self, data: pd.DataFrame, perform_hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train the XGBoost model on the provided data.

        Args:
            data: DataFrame containing features and target
            perform_hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training metrics

        Raises:
            ValueError: If required columns are missing from the data
            RuntimeError: If training fails for any other reason
        """
        logger.info(f"Training XGBoost model with {len(data)} samples")

        # Validate that all required columns are in the data
        missing_features = [col for col in self.feature_columns if col not in data.columns]
        if missing_features:
            error_msg = f"Missing required feature columns: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.target_column not in data.columns:
            error_msg = f"Target column '{self.target_column}' not found in data"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for NaN values
        if data[self.feature_columns + [self.target_column]].isna().any().any():
            warning_msg = "Data contains NaN values, which may affect model performance"
            logger.warning(warning_msg)

        try:
            # Extract features and target
            X = data[self.feature_columns]
            y = data[self.target_column]

            # Split data into training and validation sets
            val_size = self.config.get('validation_split', 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=val_size,
                random_state=self.config.get('random_state', 42)
            )

            if perform_hyperparameter_tuning:
                logger.info("Performing hyperparameter tuning...")
                self._hyperparameter_tuning(X_train, y_train, X_val, y_val)

            # Train the model
            start_time = time.time()

            logger.info("Training XGBoost model...")
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='rmse',
                early_stopping_rounds=self.config.get('early_stopping_rounds', 10),
                verbose=self.config.get('verbose', True)
            )

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")

            # Evaluate the model
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)

            # Save feature importance
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            # Save the model
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            model_filename = f"xgboost_model_{timestamp}.joblib"
            model_path = Path(self.model_path) / model_filename
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")

            # Save feature importance
            importance_path = Path(self.model_path) / f"feature_importance_{timestamp}.csv"
            feature_importance.to_csv(importance_path, index=False)

            # Return metrics
            metrics = {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'training_time': training_time,
                'feature_importance': feature_importance.to_dict(),
                'model_path': str(model_path),
                'importance_path': str(importance_path)
            }

            return metrics

        except Exception as e:
            error_msg = f"XGBoost model training failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _hyperparameter_tuning(self, X_train, y_train, X_val, y_val) -> None:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        try:
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 300],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }

            # Create XGBoost regressor
            xgb_model = xgb.XGBRegressor(objective=self.params['objective'])

            # Random search
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=self.config.get('hyperparameter_tuning_iterations', 10),
                scoring='neg_mean_squared_error',
                cv=self.config.get('cv_folds', 3),
                verbose=1,
                random_state=self.config.get('random_state', 42),
                n_jobs=-1
            )

            # Fit random search
            random_search.fit(X_train, y_train)

            # Get best parameters
            logger.info(f"Best hyperparameters: {random_search.best_params_}")

            # Update parameters with best values
            self.params.update(random_search.best_params_)

        except Exception as e:
            error_msg = f"Hyperparameter tuning failed: {str(e)}"
            logger.error(error_msg)
            logger.warning("Continuing with default parameters")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            data: DataFrame containing features

        Returns:
            Array of predictions

        Raises:
            ValueError: If the model has not been trained
            ValueError: If required feature columns are missing
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Extract features
        if isinstance(data, pd.DataFrame):
            # Check for missing columns
            missing_features = [col for col in self.feature_columns if col not in data.columns]
            if missing_features:
                error_msg = f"Missing required feature columns: {missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            X = data[self.feature_columns]
        else:
            X = data

        try:
            # Generate predictions
            predictions = self.model.predict(X)
            return predictions

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.

        Args:
            path: Path to save the model (optional)

        Returns:
            Path where the model was saved

        Raises:
            ValueError: If no model exists to save
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Generate filename with timestamp if not specified
        if path is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            path = Path(self.model_path) / f"xgboost_model_{timestamp}.joblib"
        else:
            path = Path(path)

        try:
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")

            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'params': self.params,
                'config': self.config,
                'model_type': 'XGBoost'
            }

            metadata_path = path.with_suffix('.metadata.joblib')
            joblib.dump(metadata, metadata_path)
            logger.info(f"Metadata saved to {metadata_path}")

            return str(path)

        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded XGBoostModel instance

        Raises:
            FileNotFoundError: If the model file doesn't exist
            RuntimeError: If loading fails for any reason
        """
        try:
            # Load the model
            model_path = Path(path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {path}")

            # Load metadata
            metadata_path = model_path.with_suffix('.metadata.joblib')
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                config = metadata.get('config', {})
            else:
                logger.warning(f"No metadata found for model at {path}")
                config = {}

            # Create instance
            instance = cls(config)

            # Load model
            instance.model = joblib.load(path)

            # Set attributes from metadata if available
            if metadata_path.exists():
                instance.feature_columns = metadata.get('feature_columns', instance.feature_columns)
                instance.target_column = metadata.get('target_column', instance.target_column)
                instance.params = metadata.get('params', instance.params)

            logger.info(f"Model loaded from {path}")
            return instance

        except FileNotFoundError:
            # Re-raise FileNotFoundError as is
            raise
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def train_xgboost_after_lstm(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[XGBoostModel, Dict[str, Any]]:
    """
    Train XGBoost model on data that includes LSTM predictions.

    This function is designed to be used in the two-stage modeling approach:
    1. First stage: Train LSTM model for initial predictions
    2. Second stage: Train XGBoost model using LSTM predictions as features

    Args:
        data: DataFrame containing LSTM predictions and other features
        config: Dictionary containing training configuration

    Returns:
        Tuple of (trained model, training metrics)

    Raises:
        ValueError: If the lstm_predictions column is missing
        RuntimeError: If training fails for any reason
    """
    logger.info("Training XGBoost model as second-stage bias correction")

    # Ensure the data has the LSTM predictions column
    if 'lstm_predictions' not in data.columns and config.get('feature_columns', []):
        if 'lstm_predictions' in config.get('feature_columns', []):
            error_msg = "Data is missing the lstm_predictions column required for two-stage modeling"
            logger.error(error_msg)
            raise ValueError(error_msg)

    try:
        # Create and train XGBoost model
        model = XGBoostModel(config)
        metrics = model.train(data, perform_hyperparameter_tuning=config.get('perform_hyperparameter_tuning', False))
        return model, metrics

    except Exception as e:
        error_msg = f"Failed to train XGBoost after LSTM: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def apply_xgboost_correction(data: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    Apply XGBoost bias correction to data.

    Args:
        data: DataFrame containing features for prediction
        model_path: Path to saved XGBoost model

    Returns:
        DataFrame with added XGBoost predictions

    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If data is missing required columns
        RuntimeError: If prediction fails for any reason
    """
    try:
        # Load model
        model = XGBoostModel.load(model_path)

        # Make predictions
        predictions = model.predict(data)

        # Add predictions to dataframe
        result_df = data.copy()
        result_df['xgboost_prediction'] = predictions

        # If predictions are log-transformed, add exponentiated values
        if model.target_column.startswith('log_'):
            original_column = model.target_column[4:]  # Remove 'log_' prefix
            result_df[f'xgboost_{original_column}'] = np.exp(result_df['xgboost_prediction'])

        return result_df

    except (FileNotFoundError, ValueError):
        # Re-raise these specific exceptions as they have meaningful messages
        raise
    except Exception as e:
        error_msg = f"Failed to apply XGBoost correction: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
