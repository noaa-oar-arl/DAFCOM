#!/usr/bin/env python3
'''
DAFCOM Unified Training Data Processor

This script combines observation data retrieval, model data processing, and data cleaning
into a single unified workflow for AI model training. It uses the monetio-based
observation processor and the unified model data processor to create clean, ML-ready datasets.

Author: GitHub Copilot
Date: June 5, 2025
'''

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import from DAFCOM package
from dafcom.utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules from step directories
# These imports will need to be updated when these modules are moved to the new package structure
import importlib.util
import sys

def import_module_from_path(module_name, file_path):
    """Dynamically import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not find module {module_name} at {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import the observation and model processors based on the actual location
try:
    # Paths relative to this module
    project_root = Path(__file__).parents[3]  # navigate up to the project root
    obs_processor_path = project_root / "src" / "forecast_mode" / "step2_obs_prepare" / "unified_obs_processor.py"
    data_loader_path = project_root / "src" / "forecast_mode" / "step1_input_prepare" / "modules" / "parallel_data_loader.py"

    # Import modules
    obs_module = import_module_from_path("unified_obs_processor", str(obs_processor_path))
    data_loader_module = import_module_from_path("parallel_data_loader", str(data_loader_path))

    ObservationDataLoader = obs_module.ObservationDataLoader
    ParallelDataLoader = data_loader_module.ParallelDataLoader
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure you're running from the correct directory with access to DAFCOM modules.")
    sys.exit(1)


class UnifiedTrainingDataProcessor:
    """
    Unified processor that combines observation data, model data, and performs
    data cleaning to prepare ML-ready datasets for training.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the unified processor.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        try:
            self.config = load_config(self.config_path)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

        # Extract configuration parameters
        self.start_date = self.config.get("start_date")
        self.end_date = self.config.get("end_date")
        self.model_data_dir = self.config.get("model_data_dir")
        self.obs_data_dir = self.config.get("obs_data_dir")
        self.output_dir = self.config.get("output_dir", "./output")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize sub-processors
        self._initialize_processors()

    def _initialize_processors(self):
        """Initialize the observation and model data processors."""
        try:
            # Initialize observation data processor
            obs_config = self.config.get("observation_processor", {})
            self.obs_processor = ObservationDataLoader(
                start_date=self.start_date,
                end_date=self.end_date,
                data_dir=self.obs_data_dir,
                **obs_config
            )

            # Initialize model data processor
            model_config = self.config.get("model_processor", {})
            self.model_processor = ParallelDataLoader(
                start_date=self.start_date,
                end_date=self.end_date,
                data_dir=self.model_data_dir,
                **model_config
            )

            logger.info("Processors initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise

    def process_data(self) -> pd.DataFrame:
        """
        Process both observation and model data and merge them.

        Returns:
            DataFrame containing merged and processed data
        """
        logger.info("Beginning unified data processing...")

        # Process observation data
        logger.info("Processing observation data...")
        obs_data = self.obs_processor.load_data()

        # Process model data
        logger.info("Processing model data...")
        model_data = self.model_processor.load_data()

        # Merge the datasets
        logger.info("Merging observation and model data...")
        merged_data = self._merge_datasets(obs_data, model_data)

        # Clean the merged data
        logger.info("Cleaning merged data...")
        cleaned_data = self._clean_data(merged_data)

        # Save processed data
        output_path = os.path.join(
            self.output_dir,
            f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        cleaned_data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        return cleaned_data

    def _merge_datasets(self, obs_data: pd.DataFrame, model_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge observation and model datasets.

        Args:
            obs_data: DataFrame containing observation data
            model_data: DataFrame containing model output data

        Returns:
            Merged DataFrame
        """
        # Get merge configuration
        merge_config = self.config.get("merge_settings", {})
        merge_on = merge_config.get("merge_on", ["date", "station_id"])
        how = merge_config.get("how", "inner")

        # Check for required columns
        for col in merge_on:
            if col not in obs_data.columns:
                logger.error(f"Column {col} not found in observation data")
                raise ValueError(f"Merge column {col} missing from observation data")
            if col not in model_data.columns:
                logger.error(f"Column {col} not found in model data")
                raise ValueError(f"Merge column {col} missing from model data")

        # Perform merge
        try:
            merged = pd.merge(
                obs_data, model_data,
                on=merge_on,
                how=how,
                suffixes=('_obs', '_model')
            )

            logger.info(f"Merged data shape: {merged.shape}")
            return merged

        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the merged data for ML training.

        Args:
            data: Merged DataFrame to clean

        Returns:
            Cleaned DataFrame ready for ML training
        """
        # Get data cleaning configuration
        clean_config = self.config.get("data_cleaning", {})

        # Create a copy to avoid modifying the original
        cleaned = data.copy()

        # Handle missing values
        if clean_config.get("drop_na", True):
            before_count = len(cleaned)
            cleaned = cleaned.dropna(
                subset=clean_config.get("required_columns", []),
                how=clean_config.get("na_how", "any")
            )
            after_count = len(cleaned)
            logger.info(f"Dropped {before_count - after_count} rows with missing values")

        # Fill remaining missing values if specified
        fill_method = clean_config.get("fill_method")
        if fill_method:
            if fill_method == "mean":
                cleaned = cleaned.fillna(cleaned.mean())
            elif fill_method == "median":
                cleaned = cleaned.fillna(cleaned.median())
            elif fill_method == "zero":
                cleaned = cleaned.fillna(0)

        # Remove outliers if specified
        if clean_config.get("remove_outliers", False):
            cleaned = self._remove_outliers(cleaned, clean_config)

        # Add derived features if specified
        if clean_config.get("add_derived_features", False):
            cleaned = self._add_derived_features(cleaned, clean_config)

        logger.info(f"Data cleaning completed. Final shape: {cleaned.shape}")
        return cleaned

    def _remove_outliers(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Args:
            data: DataFrame to process
            config: Configuration dictionary for outlier removal

        Returns:
            DataFrame with outliers removed
        """
        outlier_columns = config.get("outlier_columns", [])
        method = config.get("outlier_method", "zscore")
        threshold = config.get("outlier_threshold", 3.0)

        filtered_data = data.copy()

        if method == "zscore":
            for column in outlier_columns:
                if column in data.columns:
                    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                    filtered_data = filtered_data[z_scores < threshold]

        elif method == "iqr":
            for column in outlier_columns:
                if column in data.columns:
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    filtered_data = filtered_data[
                        (data[column] >= Q1 - threshold * IQR) &
                        (data[column] <= Q3 + threshold * IQR)
                    ]

        removed = len(data) - len(filtered_data)
        logger.info(f"Removed {removed} outlier rows ({removed/len(data)*100:.2f}%)")

        return filtered_data

    def _add_derived_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Add derived features to the dataset.

        Args:
            data: DataFrame to process
            config: Configuration dictionary for feature engineering

        Returns:
            DataFrame with added derived features
        """
        enhanced_data = data.copy()

        # Extract time features if date column exists
        date_col = config.get("date_column")
        if date_col and date_col in enhanced_data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(enhanced_data[date_col]):
                enhanced_data[date_col] = pd.to_datetime(enhanced_data[date_col])

            # Extract time features
            enhanced_data['hour'] = enhanced_data[date_col].dt.hour
            enhanced_data['day'] = enhanced_data[date_col].dt.day
            enhanced_data['month'] = enhanced_data[date_col].dt.month
            enhanced_data['dayofweek'] = enhanced_data[date_col].dt.dayofweek
            enhanced_data['is_weekend'] = enhanced_data['dayofweek'].isin([5, 6]).astype(int)

            # Add cyclical encoding for time features
            if config.get("use_cyclical_features", True):
                # Hour of day
                enhanced_data['hour_sin'] = np.sin(2 * np.pi * enhanced_data['hour'] / 24)
                enhanced_data['hour_cos'] = np.cos(2 * np.pi * enhanced_data['hour'] / 24)

                # Day of month
                enhanced_data['day_sin'] = np.sin(2 * np.pi * enhanced_data['day'] / 31)
                enhanced_data['day_cos'] = np.cos(2 * np.pi * enhanced_data['day'] / 31)

                # Month of year
                enhanced_data['month_sin'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
                enhanced_data['month_cos'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)

                # Day of week
                enhanced_data['dayofweek_sin'] = np.sin(2 * np.pi * enhanced_data['dayofweek'] / 7)
                enhanced_data['dayofweek_cos'] = np.cos(2 * np.pi * enhanced_data['dayofweek'] / 7)

        # Add custom derived features if specified
        custom_features = config.get("custom_features", [])
        for feature in custom_features:
            formula = feature.get("formula")
            name = feature.get("name")
            if formula and name:
                try:
                    # Use eval to calculate the formula (with safety precautions)
                    # Note: This assumes the formula is safe and only references columns in the dataframe
                    enhanced_data[name] = eval(formula, {"__builtins__": {}},
                                               {**{"df": enhanced_data},
                                                **{col: enhanced_data[col] for col in enhanced_data.columns},
                                                "np": np})
                except Exception as e:
                    logger.warning(f"Failed to create derived feature '{name}': {e}")

        logger.info(f"Added derived features. New shape: {enhanced_data.shape}")
        return enhanced_data

    def split_data(self, data: pd.DataFrame, test_size: float = 0.2,
                  val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.

        Args:
            data: DataFrame to split
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, test_data) DataFrames
        """
        # Check if temporal splitting should be used
        split_config = self.config.get("data_split", {})
        temporal_split = split_config.get("temporal_split", True)

        if temporal_split:
            # Sort data by date if date column is specified
            date_col = split_config.get("date_column")
            if date_col and date_col in data.columns:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                    data[date_col] = pd.to_datetime(data[date_col])

                # Sort by date
                data = data.sort_values(by=date_col)

            # Perform temporal split
            n_samples = len(data)
            n_test = int(n_samples * test_size)
            n_val = int(n_samples * val_size)
            n_train = n_samples - n_test - n_val

            train_data = data.iloc[:n_train].copy()
            val_data = data.iloc[n_train:n_train+n_val].copy()
            test_data = data.iloc[n_train+n_val:].copy()

        else:
            # Use random splitting
            # First split off test set
            train_val_data, test_data = train_test_split(
                data, test_size=test_size, random_state=random_state
            )

            # Then split train/val sets
            val_size_adjusted = val_size / (1 - test_size)
            train_data, val_data = train_test_split(
                train_val_data, test_size=val_size_adjusted, random_state=random_state
            )

        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        # Save the splits if specified
        if split_config.get("save_splits", False):
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            train_data.to_csv(os.path.join(self.output_dir, f"train_data_{timestamp}.csv"), index=False)
            val_data.to_csv(os.path.join(self.output_dir, f"val_data_{timestamp}.csv"), index=False)
            test_data.to_csv(os.path.join(self.output_dir, f"test_data_{timestamp}.csv"), index=False)

        return train_data, val_data, test_data

    def visualize_data(self, data: pd.DataFrame) -> None:
        """
        Visualize the processed data to help with understanding and debugging.

        Args:
            data: DataFrame to visualize
        """
        vis_config = self.config.get("visualization", {})
        if not vis_config.get("enabled", True):
            return

        logger.info("Generating data visualizations...")

        # Set up matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        fig_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        # Distribution plots for key variables
        target_cols = vis_config.get("target_columns", [])
        feature_cols = vis_config.get("feature_columns", [])

        # Combine target and selected features
        plot_columns = target_cols + feature_cols[:min(len(feature_cols), 5)]

        for col in plot_columns:
            if col in data.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(data[col].dropna(), bins=30, alpha=0.7)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save figure
                plt.savefig(os.path.join(fig_dir, f"dist_{col}.png"))
                plt.close()

        # Time series plots if date column exists
        date_col = vis_config.get("date_column")
        if date_col and date_col in data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                data[date_col] = pd.to_datetime(data[date_col])

            # Group by date if there are multiple observations per date
            if len(data) > len(data[date_col].unique()):
                daily_data = data.groupby(data[date_col].dt.date).mean().reset_index()
                date_col = 'index'
            else:
                daily_data = data.copy()

            # Plot time series for target variables
            for col in target_cols:
                if col in daily_data.columns:
                    plt.figure(figsize=(12, 6))
                    plt.plot(daily_data[date_col], daily_data[col], 'b-', linewidth=1.5)
                    plt.title(f'Time Series of {col}')
                    plt.xlabel('Date')
                    plt.ylabel(col)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # Save figure
                    plt.savefig(os.path.join(fig_dir, f"timeseries_{col}.png"))
                    plt.close()

        # Correlation heatmap
        if vis_config.get("plot_correlation", True):
            plt.figure(figsize=(12, 10))

            # Select numerical columns
            numeric_data = data.select_dtypes(include=['number'])

            # Limit to most important columns if too many
            if numeric_data.shape[1] > 15:
                cols_to_use = target_cols + feature_cols[:min(len(feature_cols), 10)]
                cols_to_use = [c for c in cols_to_use if c in numeric_data.columns]
                numeric_data = numeric_data[cols_to_use]

            corr = numeric_data.corr()

            # Plot heatmap
            import seaborn as sns
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()

            # Save figure
            plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"))
            plt.close()

        logger.info(f"Visualizations saved to {fig_dir}")

    def run_full_processing(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the full data processing pipeline.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Process the data
        processed_data = self.process_data()

        # Visualize the processed data
        self.visualize_data(processed_data)

        # Split the data
        train_data, val_data, test_data = self.split_data(processed_data)

        return train_data, val_data, test_data
