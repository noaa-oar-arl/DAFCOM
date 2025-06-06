#!/usr/bin/env python3
"""
Example script for training and using the two-stage forecasting pipeline with DAFCOM

This script demonstrates how to:
1. Process data for the two-stage pipeline
2. Train a two-stage model (LSTM + XGBoost)
3. Apply the pipeline to new data
4. Visualize the results

Author: GitHub Copilot
Date: June 6, 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from DAFCOM package
from dafcom.utils.config import load_config
from dafcom.forecast.models.two_stage_pipeline import (
    TwoStageForecastPipeline,
    train_two_stage_model,
    apply_two_stage_model
)


def main():
    """Main function to demonstrate the two-stage pipeline."""
    print("\n" + "="*80)
    print("DAFCOM Two-Stage Forecasting Pipeline Example")
    print("="*80)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DAFCOM Two-Stage Pipeline Example")
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to pipeline configuration file',
        default=None
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to training data file',
        default=None
    )
    parser.add_argument(
        '--test-data', '-t',
        type=str,
        help='Path to test data file',
        default=None
    )

    args = parser.parse_args()

    # Set paths for config and data
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[3]  # Navigate up to the project root

    if args.config is None:
        # Use default config path
        config_path = project_root / "src" / "dafcom" / "forecast" / "models" / "config" / "two_stage_pipeline_config.yaml"
        # If the config doesn't exist, create a sample one
        if not config_path.exists():
            create_sample_config(config_path)
    else:
        config_path = Path(args.config)

    # Check if training data is provided or use example data
    if args.data is None:
        # Look for example data or create synthetic data
        data_path = project_root / "src" / "dafcom" / "forecast" / "examples" / "data" / "example_training_data.csv"
        if not data_path.exists():
            data_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Creating synthetic training data at {data_path}")
            create_synthetic_data(data_path)
    else:
        data_path = Path(args.data)

    # Check if test data is provided or use example data
    if args.test_data is None:
        # Look for example test data or create synthetic test data
        test_data_path = project_root / "src" / "dafcom" / "forecast" / "examples" / "data" / "example_test_data.csv"
        if not test_data_path.exists():
            test_data_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Creating synthetic test data at {test_data_path}")
            create_synthetic_data(test_data_path, is_test=True)
    else:
        test_data_path = Path(args.test_data)

    # Train the pipeline
    train_pipeline(config_path, data_path)

    # Apply the pipeline to test data
    pipeline_path = project_root / "src" / "dafcom" / "forecast" / "examples" / "models" / "pipeline"
    if pipeline_path.exists():
        apply_pipeline(pipeline_path, test_data_path)
    else:
        print(f"Pipeline not found at {pipeline_path}. Please train the pipeline first.")


def create_sample_config(config_path):
    """Create a sample configuration file for the two-stage pipeline."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample configuration
    config_content = """# Two-Stage Pipeline Configuration

# Output directory for saving models and results
output_dir: ./models/pipeline

# Stage 1: LSTM model configuration
stage1:
  model_type: lstm
  output_dir: ./models/pipeline/stage1

  # Data preparation settings
  data_preparation:
    feature_columns:
      - v1_blh
      - v2_d2m
      - v3_e
      - v4_sp
      - v5_t2m
      - v6_tp
      - v7_u10
      - v8_v10
      - v9_aod
      - v10_luc
      - v11_elevation
      - v12_population
      - v13_ufs_pm25
      - hour_utc
      - day_of_year
    target_columns:
      - log_pm25
    time_steps: 24
    forecast_steps: 24
    scaling_method: standard

  # LSTM model settings
  model:
    lstm_units: [64, 32]
    dropout_rate: 0.2
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    early_stopping_patience: 10
    loss: mse
    metrics:
      - mae
    validation_split: 0.2

# Stage 2: XGBoost model configuration
stage2:
  output_dir: ./models/pipeline/stage2

  # Feature columns for XGBoost (includes LSTM predictions)
  feature_columns:
    - v1_blh
    - v2_d2m
    - v3_e
    - v4_sp
    - v5_t2m
    - v6_tp
    - v7_u10
    - v8_v10
    - v9_aod
    - v10_luc
    - v11_elevation
    - v12_population
    - v13_ufs_pm25
    - hour_utc
    - day_of_year
    - lstm_predictions

  # Target column for XGBoost
  target_column: log_pm25

  # XGBoost parameters
  objective: reg:squarederror
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 100
  subsample: 0.8
  colsample_bytree: 0.8
  gamma: 0
  min_child_weight: 1
  early_stopping_rounds: 10

  # Whether to perform hyperparameter tuning
  perform_hyperparameter_tuning: false
  hyperparameter_tuning_iterations: 10
  cv_folds: 3
  random_state: 42
  validation_split: 0.2
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"Sample configuration created at {config_path}")


def create_synthetic_data(file_path, is_test=False):
    """Create synthetic data for example purposes."""
    # Number of samples
    n_samples = 1000 if not is_test else 500

    # Seed for reproducibility
    np.random.seed(42 if not is_test else 43)

    # Create DataFrame with features
    data = pd.DataFrame({
        'site_index': np.random.randint(1, 100, n_samples),
        'time_utc': pd.date_range('2025-01-01', periods=n_samples, freq='H'),
        'lat': np.random.uniform(25, 50, n_samples),
        'lon': np.random.uniform(-125, -65, n_samples),
        'v1_blh': np.random.normal(1000, 500, n_samples),
        'v2_d2m': np.random.normal(15, 5, n_samples),
        'v3_e': np.random.normal(50, 10, n_samples),
        'v4_sp': np.random.normal(101325, 1000, n_samples),
        'v5_t2m': np.random.normal(288, 10, n_samples),
        'v6_tp': np.random.exponential(0.1, n_samples),
        'v7_u10': np.random.normal(0, 5, n_samples),
        'v8_v10': np.random.normal(0, 5, n_samples),
        'v9_aod': np.random.exponential(0.2, n_samples),
        'v10_luc': np.random.randint(0, 20, n_samples),
        'v11_elevation': np.random.uniform(0, 2000, n_samples),
        'v12_population': np.random.exponential(100000, n_samples),
        'v13_ufs_pm25': np.random.lognormal(2, 0.5, n_samples)
    })

    # Extract hour and day of year
    data['hour_utc'] = data['time_utc'].dt.hour
    data['day_of_year'] = data['time_utc'].dt.dayofyear

    # Create target with some relationship to the features
    pm25 = (
        0.1 * data['v1_blh'] +
        0.5 * data['v13_ufs_pm25'] +
        0.01 * data['v3_e'] -
        0.01 * data['v11_elevation'] +
        0.00001 * data['v12_population'] +
        np.random.normal(0, 5, n_samples)
    )

    # Ensure non-negative values
    pm25 = np.maximum(1.0, pm25)

    # Add target
    data['pm25'] = pm25
    data['log_pm25'] = np.log(pm25)

    # Save to CSV
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)

    print(f"Created synthetic data with {n_samples} samples at {file_path}")


def train_pipeline(config_path, data_path):
    """Train the two-stage pipeline."""
    print("\n" + "-"*40)
    print("Training Two-Stage Pipeline")
    print("-"*40)

    try:
        # Load configuration
        config = load_config(config_path)

        # Create output directory
        output_dir = Path(config.get('output_dir', './models/pipeline'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train the pipeline
        pipeline = train_two_stage_model(str(config_path), str(data_path))

        # Save the pipeline
        pipeline_dir = output_dir.parent / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save(str(pipeline_dir))

        print(f"Pipeline trained and saved to {pipeline_dir}")

    except Exception as e:
        print(f"Error training pipeline: {e}")


def apply_pipeline(pipeline_path, test_data_path):
    """Apply the trained pipeline to test data."""
    print("\n" + "-"*40)
    print("Applying Two-Stage Pipeline")
    print("-"*40)

    try:
        # Load the pipeline
        pipeline = TwoStageForecastPipeline.load(str(pipeline_path))

        # Load test data
        test_data = pd.read_csv(test_data_path)

        # Apply the pipeline
        results = apply_two_stage_model(str(pipeline_path), str(test_data_path))

        # Visualize results
        output_dir = Path(pipeline_path) / "visualizations"
        pipeline.visualize(test_data, str(output_dir))

        print(f"Pipeline applied to test data. Visualizations saved to {output_dir}")

        # Display performance metrics
        metrics_path = output_dir / "forecast_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            print("\nPerformance Metrics:")
            print(metrics_df)

    except Exception as e:
        print(f"Error applying pipeline: {e}")


if __name__ == "__main__":
    main()
