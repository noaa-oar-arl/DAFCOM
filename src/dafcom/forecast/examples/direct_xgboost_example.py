#!/usr/bin/env python3
"""
Example script for training and using XGBoost models directly with DAFCOM

This script demonstrates how to:
1. Process data for XGBoost training
2. Train a standalone XGBoost model
3. Apply the model to new data
4. Visualize the results

Author: GitHub Copilot
Date: June 14, 2025
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
from dafcom.forecast.models.xgboost_model import XGBoostModel


def main():
    """Main function to demonstrate XGBoost model training and application."""
    print("\n" + "="*80)
    print("DAFCOM XGBoost Direct Forecasting Example")
    print("="*80)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DAFCOM XGBoost Example")
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to XGBoost configuration file',
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
        config_path = project_root / "src" / "dafcom" / "forecast" / "models" / "config" / "xgboost_model_config.yaml"
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

    # Train the XGBoost model
    model_dir = train_xgboost(config_path, data_path)

    # Apply the model to test data
    apply_xgboost(model_dir, test_data_path)


def create_sample_config(config_path):
    """Create a sample configuration file for XGBoost model."""
    print(f"Creating sample XGBoost configuration at {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_content = """# XGBoost Model Configuration

# Output directory for saving models and results
output_dir: ./models/xgboost

# Feature columns for XGBoost
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

# Target column for XGBoost (use log transformed PM2.5)
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


def create_synthetic_data(file_path, is_test=False):
    """
    Create synthetic data for example purposes.

    Args:
        file_path: Path where data will be saved
        is_test: Whether this is test data (different distribution)
    """
    # Make sure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42 if not is_test else 43)

    # Number of samples
    n_samples = 5000 if not is_test else 1000

    # Create a pandas DataFrame with synthetic data
    data = {
        'v1_blh': np.random.uniform(50, 2000, n_samples),
        'v2_d2m': np.random.uniform(260, 310, n_samples),
        'v3_e': np.random.uniform(0, 50, n_samples),
        'v4_sp': np.random.uniform(900, 1100, n_samples),
        'v5_t2m': np.random.uniform(260, 310, n_samples),
        'v6_tp': np.random.uniform(0, 50, n_samples),
        'v7_u10': np.random.uniform(-20, 20, n_samples),
        'v8_v10': np.random.uniform(-20, 20, n_samples),
        'v9_aod': np.random.uniform(0, 2, n_samples),
        'v10_luc': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'v11_elevation': np.random.uniform(0, 3000, n_samples),
        'v12_population': np.random.uniform(0, 20000, n_samples),
        'v13_ufs_pm25': np.random.uniform(0, 100, n_samples),
        'hour_utc': np.random.choice(range(24), n_samples),
        'day_of_year': np.random.choice(range(1, 366), n_samples),
    }

    # Create target variable based on features
    log_pm25 = (
        0.01 * data['v1_blh'] +
        0.05 * data['v5_t2m'] -
        0.2 * data['v9_aod'] +
        0.1 * data['v13_ufs_pm25'] +
        np.random.normal(0, 0.5, n_samples)
    )

    # Add some non-linearity for test data to simulate concept drift
    if is_test:
        log_pm25 += 0.02 * data['v9_aod']**2 + 0.01 * data['v7_u10'] * data['v8_v10']

    data['log_pm25'] = log_pm25
    data['pm25'] = np.exp(log_pm25)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Synthetic {'test' if is_test else 'training'} data created with {n_samples} samples")


def train_xgboost(config_path, data_path):
    """
    Train an XGBoost model with the provided configuration and data.

    Args:
        config_path: Path to configuration file
        data_path: Path to training data file

    Returns:
        Path to the directory containing the trained model
    """
    print("\n" + "-"*40)
    print("Training XGBoost Model")
    print("-"*40)

    # Load configuration
    config = load_config(config_path)

    # Create output directory
    output_dir = Path(config.get('output_dir', './models/xgboost'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = pd.read_csv(data_path)
    print(f"Loaded data with {len(data)} samples and {len(data.columns)} columns")

    # Initialize XGBoost model
    model = XGBoostModel(config)

    # Train model
    print("Starting model training...")
    metrics = model.train(
        data,
        perform_hyperparameter_tuning=config.get('perform_hyperparameter_tuning', False)
    )

    # Print training metrics
    print("\nTraining Complete!")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Validation R²: {metrics['val_r2']:.4f}")

    # Save model to disk
    model_path = output_dir / "final_model.joblib"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Create feature importance visualization
    importance_df = pd.DataFrame({
        'Feature': metrics['feature_importance']['Feature'],
        'Importance': metrics['feature_importance']['Importance']
    })

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df.sort_values('Importance')['Feature'],
        importance_df.sort_values('Importance')['Importance']
    )
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "feature_importance.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Feature importance plot saved to {fig_path}")

    return output_dir


def apply_xgboost(model_dir, test_data_path):
    """
    Apply a trained XGBoost model to test data.

    Args:
        model_dir: Directory containing the trained model
        test_data_path: Path to test data file
    """
    print("\n" + "-"*40)
    print("Applying XGBoost Model to Test Data")
    print("-"*40)

    # Locate the model file
    model_path = Path(model_dir) / "final_model.joblib"
    if not model_path.exists():
        fallback_model = list(Path(model_dir).glob("*.joblib"))
        if fallback_model:
            model_path = fallback_model[0]
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")

    # Load model
    model = XGBoostModel.load(str(model_path))
    print(f"Model loaded from {model_path}")

    # Load test data
    test_data = pd.read_csv(test_data_path)
    print(f"Loaded test data with {len(test_data)} samples")

    # Make predictions
    predictions = model.predict(test_data)

    # Add predictions to test data
    test_data['xgboost_predictions'] = predictions

    # If predictions are log-transformed, add exponentiated values
    if model.target_column.startswith('log_'):
        original_column = model.target_column[4:]  # Remove 'log_' prefix
        test_data[f'xgboost_{original_column}'] = np.exp(test_data['xgboost_predictions'])

    # Create visualizations directory
    vis_dir = Path(model_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    target_col = model.target_column
    if target_col in test_data.columns:
        # Calculate metrics on transformed scale
        rmse = np.sqrt(mean_squared_error(test_data[target_col], predictions))
        mae = mean_absolute_error(test_data[target_col], predictions)
        r2 = r2_score(test_data[target_col], predictions)

        print(f"\nTest Set Metrics (on {target_col} scale):")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        # Create actual vs. predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(test_data[target_col], predictions, alpha=0.5)
        plt.plot(
            [test_data[target_col].min(), test_data[target_col].max()],
            [test_data[target_col].min(), test_data[target_col].max()],
            'r--'
        )
        plt.xlabel(f'Actual {target_col}')
        plt.ylabel(f'Predicted {target_col}')
        plt.title('XGBoost: Actual vs. Predicted Values')
        plt.grid(True, alpha=0.3)

        scatter_path = vis_dir / "actual_vs_predicted.png"
        plt.savefig(scatter_path)
        plt.close()
        print(f"Actual vs. predicted plot saved to {scatter_path}")

        # If using log-transform, also evaluate on original scale
        if target_col.startswith('log_'):
            original_col = target_col[4:]  # Remove 'log_' prefix
            if original_col in test_data.columns:
                actual_orig = test_data[original_col]
                pred_orig = test_data[f'xgboost_{original_col}']

                rmse_orig = np.sqrt(mean_squared_error(actual_orig, pred_orig))
                mae_orig = mean_absolute_error(actual_orig, pred_orig)
                r2_orig = r2_score(actual_orig, pred_orig)

                print(f"\nTest Set Metrics (on {original_col} scale):")
                print(f"RMSE: {rmse_orig:.4f}")
                print(f"MAE: {mae_orig:.4f}")
                print(f"R²: {r2_orig:.4f}")

                # Create actual vs. predicted plot on original scale
                plt.figure(figsize=(10, 6))
                plt.scatter(actual_orig, pred_orig, alpha=0.5)
                plt.plot(
                    [actual_orig.min(), actual_orig.max()],
                    [actual_orig.min(), actual_orig.max()],
                    'r--'
                )
                plt.xlabel(f'Actual {original_col}')
                plt.ylabel(f'Predicted {original_col}')
                plt.title('XGBoost: Actual vs. Predicted (Original Scale)')
                plt.grid(True, alpha=0.3)

                scatter_orig_path = vis_dir / "actual_vs_predicted_original_scale.png"
                plt.savefig(scatter_orig_path)
                plt.close()
                print(f"Original scale plot saved to {scatter_orig_path}")

                # Save evaluation metrics to CSV
                metrics_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²'],
                    f'{target_col} (transformed)': [rmse, mae, r2],
                    f'{original_col} (original)': [rmse_orig, mae_orig, r2_orig]
                })

                metrics_path = vis_dir / "evaluation_metrics.csv"
                metrics_df.to_csv(metrics_path, index=False)
                print(f"Evaluation metrics saved to {metrics_path}")

    # Save test results
    output_path = vis_dir / "test_predictions.csv"
    test_data.to_csv(output_path, index=False)
    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    main()
