#!/usr/bin/env python3
"""
DAFCOM Two-Stage Forecasting Pipeline Module

This module implements the two-stage forecasting approach used in DAFCOM:
1. First stage: LSTM or Transformer model for base predictions
2. Second stage: XGBoost model for further bias correction

Author: GitHub Copilot
Date: June 6, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from pathlib import Path
import matplotlib.pyplot as plt

from dafcom.utils.config import load_config
from dafcom.forecast.models.training import train_model, evaluate_model
from dafcom.forecast.models.model_factory import create_model, TimeSeriesModel
from dafcom.forecast.models.time_series_data import TimeSeriesDataPreparation
from dafcom.forecast.models.xgboost_model import XGBoostModel, train_xgboost_after_lstm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TwoStageForecastPipeline:
    """
    Two-stage forecasting pipeline that combines sequential models (LSTM or Transformer)
    with XGBoost for bias correction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the two-stage pipeline.

        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        self.stage1_model = None
        self.stage2_model = None
        self.data_prep = None

        # Set up directories
        self.output_dir = Path(config.get('output_dir', './models'))
        self.stage1_dir = self.output_dir / 'stage1'
        self.stage2_dir = self.output_dir / 'stage2'

        # Create directories
        self.stage1_dir.mkdir(parents=True, exist_ok=True)
        self.stage2_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the complete two-stage pipeline.

        Args:
            data: DataFrame containing training data

        Returns:
            Dictionary of training metrics for both stages
        """
        logger.info("Starting two-stage model training pipeline")

        # First stage: Train LSTM or Transformer model
        stage1_config = self.config.get('stage1', {})
        stage1_type = stage1_config.get('model_type', 'lstm')

        # Update output directory in stage1 config
        stage1_config['output_dir'] = str(self.stage1_dir)

        logger.info(f"Training Stage 1 ({stage1_type.upper()}) model...")
        self.stage1_model, stage1_metrics = train_model(
            stage1_config, data, model_type=stage1_type
        )

        # Generate predictions from Stage 1 model for training Stage 2
        logger.info("Generating Stage 1 predictions for Stage 2 training...")
        stage1_predictions = self._generate_stage1_predictions(data)

        # Add Stage 1 predictions to the original data
        data_for_stage2 = data.copy()
        data_for_stage2['lstm_predictions'] = stage1_predictions

        # Second stage: Train XGBoost model
        stage2_config = self.config.get('stage2', {})

        # Update output directory in stage2 config
        stage2_config['output_dir'] = str(self.stage2_dir)

        logger.info("Training Stage 2 (XGBoost) model...")
        self.stage2_model, stage2_metrics = train_xgboost_after_lstm(
            data_for_stage2, stage2_config
        )

        # Combine metrics from both stages
        metrics = {
            'stage1': stage1_metrics,
            'stage2': stage2_metrics,
            'stage1_model_path': stage1_metrics.get('model_path'),
            'stage2_model_path': stage2_metrics.get('model_path')
        }

        # Save pipeline configuration
        self._save_pipeline_config(metrics)

        logger.info("Two-stage model training completed successfully")
        return metrics

    def _generate_stage1_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the Stage 1 model for use in Stage 2 training.

        Args:
            data: DataFrame containing features

        Returns:
            Array of predictions from Stage 1 model
        """
        if self.stage1_model is None:
            raise ValueError("Stage 1 model has not been trained yet")

        # Extract necessary data based on the model's configuration
        stage1_config = self.config.get('stage1', {})
        data_prep_config = stage1_config.get('data_preparation', {})

        # Initialize data preparation if not already done
        if self.data_prep is None:
            self.data_prep = TimeSeriesDataPreparation(data_prep_config)
            self.data_prep.fit_scalers(data)

        # Transform data
        transformed_data = self.data_prep.transform_data(data)

        # Create sequences
        X, _ = self.data_prep.create_sequences(transformed_data)

        # Generate predictions
        scaled_predictions = self.stage1_model.predict(X)

        # Inverse transform predictions
        predictions = self.data_prep.inverse_transform_targets(scaled_predictions)

        # Reshape predictions to match the original data length
        # Note: This depends on the specific sequence handling of your implementation
        # Adjust as needed to match how the sequences were created
        sequence_length = data_prep_config.get('time_steps', 96)
        forecast_horizon = data_prep_config.get('forecast_steps', 24)

        # Taking the first step of each forecast
        flattened_predictions = []
        for i in range(len(predictions)):
            flattened_predictions.append(predictions[i, 0, 0])

        # Pad with NaN for the first sequence_length items
        padded_predictions = [np.nan] * sequence_length + flattened_predictions
        padded_predictions = padded_predictions[:len(data)]

        return np.array(padded_predictions)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the complete pipeline.

        Args:
            data: DataFrame containing features

        Returns:
            DataFrame with added prediction columns
        """
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Models have not been trained or loaded yet")

        # Generate Stage 1 predictions
        stage1_predictions = self._generate_stage1_predictions(data)

        # Add Stage 1 predictions to the data
        data_for_stage2 = data.copy()
        data_for_stage2['lstm_predictions'] = stage1_predictions

        # Generate Stage 2 predictions
        stage2_predictions = self.stage2_model.predict(data_for_stage2)

        # Add predictions to the result DataFrame
        result_df = data.copy()
        result_df['stage1_predictions'] = stage1_predictions
        result_df['stage2_predictions'] = stage2_predictions

        # If predictions are log-transformed, add exponentiated values
        if self.config.get('stage2', {}).get('target_column', '').startswith('log_'):
            original_column = self.config.get('stage2', {}).get('target_column')[4:]
            result_df[f'final_{original_column}'] = np.exp(result_df['stage2_predictions'])
        else:
            result_df['final_predictions'] = result_df['stage2_predictions']

        return result_df

    def save(self, path: Optional[str] = None) -> Dict[str, str]:
        """
        Save the complete pipeline.

        Args:
            path: Directory to save the pipeline (optional)

        Returns:
            Dictionary with paths to saved models
        """
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Models have not been trained yet")

        # Generate path with timestamp if not specified
        if path is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            path = self.output_dir / f"pipeline_{timestamp}"
        else:
            path = Path(path)

        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)

        # Save Stage 1 model
        stage1_path = path / "stage1_model.h5"
        self.stage1_model.save(str(stage1_path))

        # Save Stage 2 model
        stage2_path = path / "stage2_model.joblib"
        self.stage2_model.save(str(stage2_path))

        # Save scalers if available
        scaler_path = None
        if self.data_prep is not None:
            scaler_path = path / "scalers"
            self.data_prep.save_scalers(str(scaler_path))

        # Save pipeline configuration
        config_path = path / "pipeline_config.joblib"
        import joblib
        joblib.dump(self.config, config_path)

        paths = {
            'pipeline_dir': str(path),
            'stage1_model': str(stage1_path),
            'stage2_model': str(stage2_path),
            'config': str(config_path)
        }

        if scaler_path is not None:
            paths['scalers'] = str(scaler_path)

        logger.info(f"Pipeline saved to {path}")
        return paths

    @classmethod
    def load(cls, path: str) -> 'TwoStageForecastPipeline':
        """
        Load a saved pipeline.

        Args:
            path: Path to the saved pipeline directory

        Returns:
            Loaded pipeline instance
        """
        path = Path(path)

        # Load pipeline configuration
        import joblib
        config_path = path / "pipeline_config.joblib"
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline configuration not found at {config_path}")

        config = joblib.load(config_path)

        # Create pipeline instance
        pipeline = cls(config)

        # Load Stage 1 model
        from dafcom.forecast.models.model_factory import LSTMModel, TransformerModel
        stage1_path = path / "stage1_model.h5"
        metadata_path = stage1_path.with_name(f"{stage1_path.stem}_metadata.joblib")

        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            model_type = metadata.get('model_type', '').lower()

            if model_type == 'lstmmodel':
                pipeline.stage1_model = LSTMModel.load(str(stage1_path))
            elif model_type == 'transformermodel':
                pipeline.stage1_model = TransformerModel.load(str(stage1_path))
            else:
                raise ValueError(f"Unsupported Stage 1 model type: {model_type}")
        else:
            raise FileNotFoundError(f"Stage 1 model metadata not found at {metadata_path}")

        # Load Stage 2 model
        stage2_path = path / "stage2_model.joblib"
        pipeline.stage2_model = XGBoostModel.load(str(stage2_path))

        # Load scalers if available
        scaler_path = path / "scalers"
        if scaler_path.exists():
            stage1_config = config.get('stage1', {})
            data_prep_config = stage1_config.get('data_preparation', {})
            pipeline.data_prep = TimeSeriesDataPreparation(data_prep_config)
            pipeline.data_prep.load_scalers(str(scaler_path))

        logger.info(f"Pipeline loaded from {path}")
        return pipeline

    def _save_pipeline_config(self, metrics: Dict[str, Any]) -> None:
        """
        Save the pipeline configuration and metrics.

        Args:
            metrics: Dictionary containing training metrics
        """
        # Combine config and metrics
        pipeline_info = {
            'config': self.config,
            'metrics': metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to file
        import json
        info_path = self.output_dir / "pipeline_info.json"

        with open(info_path, 'w') as f:
            json.dump(pipeline_info, f, indent=2)

        logger.info(f"Pipeline information saved to {info_path}")

    def visualize(self, test_data: pd.DataFrame, output_dir: Optional[str] = None) -> None:
        """
        Visualize the performance of the pipeline on test data.

        Args:
            test_data: DataFrame containing test data
            output_dir: Directory to save visualizations (optional)
        """
        if output_dir is None:
            output_dir = self.output_dir / "visualizations"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate predictions
        predictions_df = self.predict(test_data)

        # Extract actual values and predictions
        target_col = self.config.get('stage2', {}).get('target_column', 'pm25')
        if target_col.startswith('log_'):
            actual_col = target_col[4:]  # Remove 'log_' prefix
            actual_values = np.exp(test_data[target_col])
            final_predictions = predictions_df[f'final_{actual_col}']
        else:
            actual_col = target_col
            actual_values = test_data[target_col]
            final_predictions = predictions_df['final_predictions']

        stage1_predictions = predictions_df['stage1_predictions']

        # Plot time series
        plt.figure(figsize=(12, 6))
        plt.plot(actual_values, 'b-', label='Actual')
        plt.plot(stage1_predictions, 'g--', label='Stage 1 (LSTM)')
        plt.plot(final_predictions, 'r-', label='Stage 2 (XGBoost Corrected)')
        plt.title(f'Two-Stage Forecast for {actual_col}')
        plt.xlabel('Time Step')
        plt.ylabel(actual_col)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(output_dir / "forecast_comparison.png")
        plt.close()

        # Plot scatter comparison
        plt.figure(figsize=(14, 6))

        # Stage 1 comparison
        plt.subplot(1, 2, 1)
        plt.scatter(actual_values, stage1_predictions, alpha=0.5)
        plt.plot([actual_values.min(), actual_values.max()],
                [actual_values.min(), actual_values.max()], 'k--')
        plt.title('Stage 1 (LSTM) vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Stage 1 Predictions')
        plt.grid(True, alpha=0.3)

        # Stage 2 comparison
        plt.subplot(1, 2, 2)
        plt.scatter(actual_values, final_predictions, alpha=0.5)
        plt.plot([actual_values.min(), actual_values.max()],
                [actual_values.min(), actual_values.max()], 'k--')
        plt.title('Stage 2 (XGBoost Corrected) vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Stage 2 Predictions')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plt.savefig(output_dir / "prediction_comparison.png")
        plt.close()

        # Calculate and display metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Create a metrics dataframe
        metrics = {
            'Model': ['Stage 1 (LSTM)', 'Stage 2 (XGBoost Corrected)'],
            'RMSE': [
                np.sqrt(mean_squared_error(actual_values, stage1_predictions)),
                np.sqrt(mean_squared_error(actual_values, final_predictions))
            ],
            'MAE': [
                mean_absolute_error(actual_values, stage1_predictions),
                mean_absolute_error(actual_values, final_predictions)
            ],
            'R²': [
                r2_score(actual_values, stage1_predictions),
                r2_score(actual_values, final_predictions)
            ]
        }

        metrics_df = pd.DataFrame(metrics)

        # Save metrics
        metrics_df.to_csv(output_dir / "forecast_metrics.csv", index=False)

        logger.info(f"Visualizations saved to {output_dir}")
        logger.info(f"Performance metrics:\n{metrics_df}")


def train_two_stage_model(config_path: str, data_path: str) -> TwoStageForecastPipeline:
    """
    Train a complete two-stage forecasting pipeline.

    Args:
        config_path: Path to pipeline configuration file
        data_path: Path to training data file

    Returns:
        Trained pipeline instance
    """
    # Load configuration
    config = load_config(config_path)

    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded training data with {len(data)} rows and {len(data.columns)} columns")

    # Create and train pipeline
    pipeline = TwoStageForecastPipeline(config)
    metrics = pipeline.train(data)

    logger.info("Two-stage model training completed successfully")
    return pipeline


def apply_two_stage_model(pipeline_path: str, data_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Apply a trained two-stage model to new data.

    Args:
        pipeline_path: Path to saved pipeline
        data_path: Path to input data file
        output_path: Path to save output data (optional)

    Returns:
        DataFrame with predictions
    """
    # Load pipeline
    pipeline = TwoStageForecastPipeline.load(pipeline_path)

    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded input data with {len(data)} rows and {len(data.columns)} columns")

    # Generate predictions
    result_df = pipeline.predict(data)

    # Save results if path provided
    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return result_df
