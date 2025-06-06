#!/usr/bin/env python3
"""
DAFCOM Time Series Forecasting Model Training Module

This module provides a unified interface for training time series forecasting models,
with support for both LSTM and Transformer architectures. It integrates with the
unified training data processor to prepare data and supports efficient training
using Dask for distributed processing.

Author: GitHub Copilot
Date: June 5, 2025
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json
import time
import matplotlib.pyplot as plt

# Import from DAFCOM package
from dafcom.utils.config import load_config
from dafcom.forecast.models.model_factory import create_model, TimeSeriesModel
from dafcom.forecast.models.time_series_data import TimeSeriesDataPreparation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dafcom.model.training")


def train_model(config: Dict[str, Any], data: pd.DataFrame,
                model_type: str = "lstm") -> Tuple[TimeSeriesModel, Dict[str, Any]]:
    """
    Train a time series forecasting model with the provided data.

    Args:
        config: Dictionary containing model and training configuration
        data: DataFrame containing time series data
        model_type: Type of model to train ('lstm' or 'transformer')

    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    logger.info(f"Initializing {model_type} model training...")

    # Initialize data preparation
    data_prep_config = config.get("data_preparation", {})
    data_prep = TimeSeriesDataPreparation(data_prep_config)

    # Process the data
    logger.info("Preparing time series data...")

    # Fit scalers and transform data
    data_prep.fit_scalers(data)
    transformed_data = data_prep.transform_data(data)

    # Create input-output sequences
    X, y = data_prep.create_sequences(transformed_data)

    # Split into training and validation sets
    train_size = int(len(X) * (1 - config.get("validation_split", 0.2)))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    # Create and build model
    model_config = config.get("model", {})

    # Add data dimensions to model config
    model_config["feature_columns"] = data_prep_config.get("feature_columns", [])
    model_config["target_columns"] = data_prep_config.get("target_columns", [])
    model_config["sequence_length"] = data_prep_config.get("time_steps", 96)
    model_config["forecast_horizon"] = data_prep_config.get("forecast_steps", 24)

    # Create model
    model = create_model(model_type, model_config)

    # Train the model
    start_time = time.time()
    history = model.fit(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Save model and scalers
    model_dir = config.get("output_dir", "./models")
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_type}_model_{timestamp}.h5")
    model.save(model_path)

    scaler_dir = os.path.join(model_dir, f"scalers_{timestamp}")
    data_prep.save_scalers(scaler_dir)

    # Calculate and save metrics
    metrics = {
        "model_type": model_type,
        "training_time": training_time,
        "train_loss": history.get("loss", [])[-1] if history.get("loss") else None,
        "val_loss": history.get("val_loss", [])[-1] if history.get("val_loss") else None,
        "train_mae": history.get("mae", [])[-1] if history.get("mae") else None,
        "val_mae": history.get("val_mae", [])[-1] if history.get("val_mae") else None,
        "model_path": model_path,
        "scaler_path": scaler_dir,
        "data_shape": {
            "n_samples": len(data),
            "n_features": len(data_prep_config.get("feature_columns", [])),
            "n_targets": len(data_prep_config.get("target_columns", [])),
            "sequence_length": data_prep_config.get("time_steps", 96),
            "forecast_horizon": data_prep_config.get("forecast_steps", 24)
        },
        "timestamp": timestamp
    }

    # Save metrics to JSON
    metrics_path = os.path.join(model_dir, f"{model_type}_metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")

    return model, metrics


def evaluate_model(model: TimeSeriesModel,
                  data_prep: TimeSeriesDataPreparation,
                  test_data: pd.DataFrame,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained time series model
        data_prep: Data preparation object with fitted scalers
        test_data: DataFrame containing test data
        config: Dictionary containing evaluation configuration

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test data...")

    # Transform test data
    transformed_test_data = data_prep.transform_data(test_data)

    # Create test sequences
    X_test, y_test = data_prep.create_sequences(transformed_test_data)

    # Make predictions
    logger.info(f"Making predictions on test data of shape {X_test.shape}")
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred_original = data_prep.inverse_transform_targets(y_pred)
    y_test_original = data_prep.inverse_transform_targets(y_test)

    # Calculate metrics
    mae = np.mean(np.abs(y_pred_original - y_test_original))
    mse = np.mean((y_pred_original - y_test_original) ** 2)
    rmse = np.sqrt(mse)

    metrics = {
        "test_mae": float(mae),
        "test_mse": float(mse),
        "test_rmse": float(rmse),
        "test_samples": len(X_test)
    }

    logger.info(f"Evaluation metrics: MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Optionally plot predictions
    if config.get("plot_predictions", False):
        plot_dir = config.get("plot_dir", "./plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Get sample indices to plot
        n_samples = min(config.get("n_plot_samples", 5), len(X_test))
        sample_indices = np.linspace(0, len(X_test)-1, n_samples, dtype=int)

        for idx in sample_indices:
            plot_prediction(y_test_original[idx], y_pred_original[idx],
                           data_prep.target_columns, idx, plot_dir)

    return metrics


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray,
                   target_columns: list, sample_idx: int,
                   output_dir: str) -> None:
    """
    Plot actual vs predicted values for a single sample.

    Args:
        y_true: True values (shape: [forecast_steps, n_targets])
        y_pred: Predicted values (shape: [forecast_steps, n_targets])
        target_columns: Names of target columns
        sample_idx: Index of the sample (for naming the file)
        output_dir: Directory to save the plots
    """
    n_targets = len(target_columns)
    forecast_steps = y_true.shape[0]
    x_axis = np.arange(forecast_steps)

    # Create figure with subplots for each target variable
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 3*n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]  # Make sure axes is always a list

    for i in range(n_targets):
        ax = axes[i]
        target_name = target_columns[i]

        # Plot actual and predicted values
        ax.plot(x_axis, y_true[:, i], 'b-', label='Actual')
        ax.plot(x_axis, y_pred[:, i], 'r--', label='Predicted')

        ax.set_title(f'{target_name} - Forecast')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"forecast_sample_{sample_idx}_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close(fig)

    logger.info(f"Prediction plot saved to {plot_path}")


def load_model_with_data_prep(model_path: str, scaler_path: str) -> Tuple[TimeSeriesModel, TimeSeriesDataPreparation]:
    """
    Load a trained model and its data preparation object.

    Args:
        model_path: Path to the saved model file
        scaler_path: Path to the directory containing saved scalers

    Returns:
        Tuple of (loaded model, data preparation object)
    """
    # Load model metadata to determine model type
    metadata_path = model_path.replace('.h5', '_metadata.joblib')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

    import joblib
    metadata = joblib.load(metadata_path)
    model_type = metadata.get('model_type', '').lower()

    # Load model using the factory
    if model_type == 'lstmmodel':
        from dafcom.forecast.models.model_factory import LSTMModel
        model = LSTMModel.load(model_path)
    elif model_type == 'transformermodel':
        from dafcom.forecast.models.model_factory import TransformerModel
        model = TransformerModel.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Initialize data preparation with config from model metadata
    data_prep_config = {
        "feature_columns": metadata.get('feature_columns', []),
        "target_columns": metadata.get('target_columns', []),
        "time_steps": metadata.get('sequence_length', 96),
        "forecast_steps": metadata.get('forecast_horizon', 24)
    }

    data_prep = TimeSeriesDataPreparation(data_prep_config)

    # Load scalers
    data_prep.load_scalers(scaler_path)

    return model, data_prep
