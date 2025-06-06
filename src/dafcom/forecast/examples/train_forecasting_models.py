#!/usr/bin/env python3
"""
Example script for training time series forecasting models with DAFCOM

This script demonstrates how to:
1. Train an LSTM model for time series forecasting
2. Train a Transformer model for time series forecasting
3. Compare their performance

Author: GitHub Copilot
Date: June 5, 2025
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from DAFCOM package
from dafcom.utils.config import load_config
from dafcom.forecast.models.training import train_model, evaluate_model
from dafcom.forecast.models import LSTMModel, TransformerModel, TimeSeriesDataPreparation


def train_both_models(data_path=None):
    """Train both LSTM and Transformer models and compare results."""
    print("\n" + "="*80)
    print("DAFCOM Time Series Forecasting Model Training Example")
    print("="*80)

    # Get paths for configs
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[3]  # Navigate up to the project root

    lstm_config_path = project_root / "src" / "forecast_mode" / "ml_models" / "config" / "lstm_model_config.yaml"
    transformer_config_path = project_root / "src" / "forecast_mode" / "ml_models" / "config" / "transformer_model_config.yaml"

    if not lstm_config_path.exists() or not transformer_config_path.exists():
        print("Error: Configuration files not found. Please check the paths.")
        return

    # Set the data path
    if data_path is None:
        # Use default example data
        data_path = project_root / "src" / "forecast_mode" / "ml_models" / "processed_data" / "example_training_data.csv"
        if not data_path.exists():
            print(f"Error: Example data not found at {data_path}")
            return

    # Load data
    import pandas as pd
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Train LSTM model
    print("\n" + "-"*40)
    print("Training LSTM Model")
    print("-"*40)

    lstm_config = load_config(lstm_config_path)
    # Add output directory to save models and results
    lstm_config["output_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "lstm")
    os.makedirs(lstm_config["output_dir"], exist_ok=True)

    lstm_model, lstm_metrics = train_model(lstm_config, data, model_type="lstm")

    # Train Transformer model
    print("\n" + "-"*40)
    print("Training Transformer Model")
    print("-"*40)

    transformer_config = load_config(transformer_config_path)
    # Add output directory to save models and results
    transformer_config["output_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "transformer")
    os.makedirs(transformer_config["output_dir"], exist_ok=True)

    transformer_model, transformer_metrics = train_model(transformer_config, data, model_type="transformer")

    # Compare models
    print("\n" + "-"*40)
    print("Model Performance Comparison")
    print("-"*40)

    print(f"LSTM Model:")
    print(f"  Training Loss: {lstm_metrics['train_loss']:.6f}")
    print(f"  Validation Loss: {lstm_metrics['val_loss']:.6f}")
    print(f"  Training MAE: {lstm_metrics['train_mae']:.6f}")
    print(f"  Validation MAE: {lstm_metrics['val_mae']:.6f}")

    print(f"\nTransformer Model:")
    print(f"  Training Loss: {transformer_metrics['train_loss']:.6f}")
    print(f"  Validation Loss: {transformer_metrics['val_loss']:.6f}")
    print(f"  Training MAE: {transformer_metrics['train_mae']:.6f}")
    print(f"  Validation MAE: {transformer_metrics['val_mae']:.6f}")

    # Plot training history comparison
    plot_comparison(lstm_model, transformer_model)

    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)


def plot_comparison(lstm_model, transformer_model):
    """Plot and compare training history of both models."""
    plt.figure(figsize=(12, 10))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(lstm_model.history['loss'], label='LSTM Training Loss')
    plt.plot(lstm_model.history['val_loss'], label='LSTM Validation Loss')
    plt.plot(transformer_model.history['loss'], label='Transformer Training Loss')
    plt.plot(transformer_model.history['val_loss'], label='Transformer Validation Loss')
    plt.title('Model Loss Comparison')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot MAE
    plt.subplot(2, 1, 2)
    plt.plot(lstm_model.history['mae'], label='LSTM Training MAE')
    plt.plot(lstm_model.history['val_mae'], label='LSTM Validation MAE')
    plt.plot(transformer_model.history['mae'], label='Transformer Training MAE')
    plt.plot(transformer_model.history['val_mae'], label='Transformer Validation MAE')
    plt.title('Model MAE Comparison')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()

    # Save the comparison plot
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()

    print(f"Comparison plot saved to {os.path.join(output_dir, 'model_comparison.png')}")


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description='DAFCOM Time Series Model Training Example')
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to CSV data file for training'
    )

    args = parser.parse_args()
    train_both_models(args.data)


if __name__ == "__main__":
    main()
