#!/usr/bin/env python3
"""
DAFCOM - Forecast    # Train model command
    train_parser = subparsers.add_parser("train", help="Train a forecasting model")
    train_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to model training configuration file"
    )
    train_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to processed data file (CSV)"
    )
    train_parser.add_argument(
        "--model-type", "-m",
        choices=["lstm", "transformer", "xgboost", "two-stage"],
        default="lstm",
        help="Type of model to train"
    )s script provides a command-line interface for training and running
DAFCOM forecasting models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from dafcom.utils.config import load_config
from dafcom.forecast.processor import UnifiedTrainingDataProcessor
from dafcom.forecast.models.training import train_model, evaluate_model
from dafcom.forecast.models.xgboost_model import XGBoostModel, train_xgboost_after_lstm
from dafcom.forecast.models.two_stage_pipeline import (
    TwoStageForecastPipeline, train_two_stage_model, apply_two_stage_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dafcom.cli")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DAFCOM Forecasting Model CLI")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Process data command
    process_parser = subparsers.add_parser("process", help="Process data for model training")
    process_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to data processing configuration file"
    )

    # Train model command
    train_parser = subparsers.add_parser("train", help="Train a forecasting model")
    train_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to model training configuration file"
    )
    train_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to processed data file (CSV)"
    )
    train_parser.add_argument(
        "--model-type", "-m",
        choices=["lstm", "transformer"],
        default="lstm",
        help="Type of model to train"
    )

    # Evaluate model command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to saved model file"
    )
    eval_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to test data file (CSV)"
    )
    eval_parser.add_argument(
        "--scalers", "-s",
        required=True,
        help="Path to directory containing model scalers"
    )
    eval_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to evaluation configuration file"
    )

    return parser.parse_args()


def process_command(args):
    """Process data for model training."""
    logger.info(f"Processing data using config: {args.config}")

    try:
        processor = UnifiedTrainingDataProcessor(args.config)
        train_data, val_data, test_data = processor.run_full_processing()

        logger.info("Data processing completed successfully")
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Validation data shape: {val_data.shape}")
        logger.info(f"Test data shape: {test_data.shape}")

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        sys.exit(1)


def train_command(args):
    """Train a forecasting model."""
    logger.info(f"Training {args.model_type} model using data: {args.data}")

    try:
        # Load data
        import pandas as pd
        data = pd.read_csv(args.data)

        # Load configuration
        config = load_config(args.config)

        # Train model based on type
        if args.model_type in ["lstm", "transformer"]:
            # Train sequence model
            model, metrics = train_model(config, data, args.model_type)

            logger.info("Model training completed successfully")
            logger.info(f"Model saved to: {metrics['model_path']}")
            logger.info(f"Training metrics: Loss={metrics['train_loss']:.6f}, MAE={metrics['train_mae']:.6f}")
            logger.info(f"Validation metrics: Loss={metrics['val_loss']:.6f}, MAE={metrics['val_mae']:.6f}")

        elif args.model_type == "xgboost":
            # Train XGBoost model
            model = XGBoostModel(config)
            metrics = model.train(data, perform_hyperparameter_tuning=config.get('perform_hyperparameter_tuning', False))

            logger.info("XGBoost model training completed successfully")
            logger.info(f"Model saved to: {metrics['model_path']}")
            logger.info(f"Training metrics: RMSE={metrics['train_rmse']:.6f}, MAE={metrics['train_mae']:.6f}")
            logger.info(f"Validation metrics: RMSE={metrics['val_rmse']:.6f}, MAE={metrics['val_mae']:.6f}")

        elif args.model_type == "two-stage":
            # Train two-stage pipeline
            pipeline = TwoStageForecastPipeline(config)
            metrics = pipeline.train(data)

            logger.info("Two-stage model training completed successfully")
            logger.info(f"Stage 1 model saved to: {metrics['stage1_model_path']}")
            logger.info(f"Stage 2 model saved to: {metrics['stage2_model_path']}")
            logger.info("Stage 1 metrics:")
            logger.info(f"  Training Loss={metrics['stage1']['train_loss']:.6f}, MAE={metrics['stage1']['train_mae']:.6f}")
            logger.info(f"  Validation Loss={metrics['stage1']['val_loss']:.6f}, MAE={metrics['stage1']['val_mae']:.6f}")
            logger.info("Stage 2 metrics:")
            logger.info(f"  Training RMSE={metrics['stage2']['train_rmse']:.6f}, MAE={metrics['stage2']['train_mae']:.6f}")
            logger.info(f"  Validation RMSE={metrics['stage2']['val_rmse']:.6f}, MAE={metrics['stage2']['val_mae']:.6f}")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)


def evaluate_command(args):
    """Evaluate a trained model."""
    logger.info(f"Evaluating model: {args.model}")

    try:
        # Load data
        import pandas as pd
        data = pd.read_csv(args.data)

        # Load configuration
        config = load_config(args.config)

        # Load model and data preparation
        from dafcom.forecast.models.training import load_model_with_data_prep
        model, data_prep = load_model_with_data_prep(args.model, args.scalers)

        # Evaluate model
        metrics = evaluate_model(model, data_prep, data, config)

        logger.info("Model evaluation completed successfully")
        logger.info(f"Test MAE: {metrics['test_mae']:.6f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.6f}")

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()

    if args.command == "process":
        process_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
