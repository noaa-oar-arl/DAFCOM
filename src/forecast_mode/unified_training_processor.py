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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parents[1] / "step2_obs_prepare"))
sys.path.append(str(Path(__file__).parents[1] / "step1_input_prepare"))

# Import the observation and model processors
try:
    from unified_obs_processor import ObservationDataLoader
    from modules.parallel_data_loader import ParallelDataLoader
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
        Initialize the unified training data processor.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Configure logging
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO').upper())
        logging.getLogger().setLevel(log_level)

        # Set output directories
        self.output_dir = Path(self.config.get('output', {}).get('output_dir', './processed_training_data/'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize observation and model data loaders
        self.obs_loader = ObservationDataLoader(self.config.get('paths', {}).get('obs_config'))
        self.model_loader = ParallelDataLoader(self.config.get('paths', {}).get('model_config'))

        logger.info(f"UnifiedTrainingDataProcessor initialized with config from {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def get_observation_data(self,
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime],
                           species: List[str] = None,
                           bbox: List[float] = None) -> xr.Dataset:
        """
        Retrieve and process observation data.

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            species: List of species to retrieve (default from config)
            bbox: Bounding box [lat_min, lon_min, lat_max, lon_max] (default from config)

        Returns:
            xarray.Dataset with the processed observation data
        """
        # Get defaults from config if not provided
        if species is None:
            species = self.config.get('observations', {}).get('species', ['PM25'])

        if bbox is None:
            bbox_config = self.config.get('grid', {}).get('bbox', {})
            bbox = [
                bbox_config.get('lat_min', 25),
                bbox_config.get('lon_min', -125),
                bbox_config.get('lat_max', 49),
                bbox_config.get('lon_max', -65)
            ]

        # Get list of observation sources from config
        sources = self.config.get('observations', {}).get('sources', ['airnow'])

        # Get local file paths if specified
        local_files = self.config.get('observations', {}).get('local_files', {})

        # Process observations
        logger.info(f"Retrieving observation data for {species} from {sources}")
        obs_data = self.obs_loader.process_observations(
            start_date=start_date,
            end_date=end_date,
            species=species,
            sources=sources,
            bbox=bbox,
            local_files=local_files
        )

        logger.info(f"Retrieved observation data with {len(obs_data.time) if 'time' in obs_data.dims else 0} timesteps")
        return obs_data

    def get_model_data(self,
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime],
                     categories: List[str] = None,
                     variables: List[str] = None) -> xr.Dataset:
        """
        Retrieve and process model forecast data.

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            categories: List of categories to retrieve (meteorology, chemistry, aerosol)
            variables: List of specific variables to retrieve

        Returns:
            xarray.Dataset with the processed model data
        """
        # Update the model loader configuration for the specified dates
        date_config = {
            'input': {
                'time_range': {
                    'start_date': start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date,
                    'end_date': end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
                }
            }
        }

        # Process model data using the unified atmospheric data processor
        if categories:
            # Process specific categories
            category_datasets = {}
            for category in categories:
                logger.info(f"Processing model data for category: {category}")
                category_datasets[category] = self.model_loader.process_atmospheric_data(category=category)

            # Merge all category datasets
            model_data = xr.merge([ds for ds in category_datasets.values()])

        elif variables:
            # Process specific variables
            logger.info(f"Processing model data for variables: {variables}")
            model_data = self.model_loader.process_atmospheric_data(variables=variables)

        else:
            # Process all data
            logger.info("Processing all model data")
            model_data = self.model_loader.process_atmospheric_data()

        logger.info(f"Retrieved model data with {len(model_data.time) if 'time' in model_data.dims else 0} timesteps")
        return model_data

    def interpolate_model_to_obs_locations(self, model_data: xr.Dataset, obs_data: xr.Dataset) -> pd.DataFrame:
        """
        Interpolate model data to observation locations.

        Args:
            model_data: Model forecast data as xarray Dataset
            obs_data: Observation data as xarray Dataset

        Returns:
            pandas DataFrame with model values at observation locations
        """
        logger.info("Interpolating model data to observation locations")

        # Get coordinates from observation data
        lat_var = 'latitude' if 'latitude' in obs_data.variables else 'lat'
        lon_var = 'longitude' if 'longitude' in obs_data.variables else 'lon'
        time_var = 'time' if 'time' in obs_data.dims else 'datetime'

        # Convert observations to DataFrame for easier processing
        obs_df = obs_data.to_dataframe().reset_index()

        # Initialize list for interpolated results
        interpolated_data = []

        # Process in chunks to avoid memory issues
        chunk_size = self.config.get('processing', {}).get('chunk_size', 1000)
        total_rows = len(obs_df)

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            logger.info(f"Processing observations {start_idx} to {end_idx} of {total_rows}")

            # Process this chunk of observations
            chunk_df = obs_df.iloc[start_idx:end_idx].copy()

            # Interpolate each variable from the model data to the obs locations
            for var_name in model_data.data_vars:
                # Get the model variable data array
                var_data = model_data[var_name]

                # Skip if dimensions don't match expectations
                if not all(dim in var_data.dims for dim in ['time', 'lat', 'lon']):
                    logger.warning(f"Skipping variable {var_name} - unexpected dimensions: {var_data.dims}")
                    continue

                # For each observation point in this chunk
                var_values = []
                for _, row in chunk_df.iterrows():
                    # Get the coordinates
                    lat = row[lat_var]
                    lon = row[lon_var]
                    time = row[time_var]

                    try:
                        # Find nearest time in model data
                        time_idx = abs(model_data.time.values - np.datetime64(time)).argmin()
                        model_time = model_data.time.values[time_idx]

                        # Interpolate to the observation location
                        value = var_data.sel(
                            time=model_time,
                            lat=lat,
                            lon=lon,
                            method='nearest'
                        ).values

                        var_values.append(float(value))
                    except Exception as e:
                        logger.debug(f"Error interpolating {var_name} at lat={lat}, lon={lon}, time={time}: {e}")
                        var_values.append(np.nan)

                # Add the interpolated values to the chunk dataframe
                chunk_df[f"model_{var_name}"] = var_values

            # Append this chunk to the results
            interpolated_data.append(chunk_df)

        # Combine all chunks
        result_df = pd.concat(interpolated_data, ignore_index=True)
        logger.info(f"Completed interpolation for {len(result_df)} points")

        return result_df

    def clean_merged_data(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the merged observation and model data.

        Args:
            merged_df: DataFrame with merged observation and model data

        Returns:
            Cleaned DataFrame ready for ML training
        """
        logger.info("Cleaning merged data for ML training")

        # Make a copy to avoid modifying the original
        df = merged_df.copy()

        # 1. Drop rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")

        # 2. Apply value range filters from config
        valid_ranges = self.config.get('quality_control', {}).get('valid_ranges', {})

        for var, (min_val, max_val) in valid_ranges.items():
            if var in df.columns:
                valid_mask = (df[var] >= min_val) & (df[var] <= max_val)
                df = df[valid_mask]
                logger.info(f"Filtered {var} to range [{min_val}, {max_val}], kept {len(df)} rows")

        # 3. Drop duplicates based on location and time
        # Find appropriate column names
        site_col = None
        time_col = None

        possible_site_cols = ['site_index', 'site_id', 'site', 'station']
        for col in possible_site_cols:
            if col in df.columns:
                site_col = col
                break

        possible_time_cols = ['time_utc', 'time', 'datetime']
        for col in possible_time_cols:
            if col in df.columns:
                time_col = col
                break

        if site_col and time_col:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=[site_col, time_col], keep='first')
            logger.info(f"Removed {before_dedup - len(df)} duplicate rows based on {site_col} and {time_col}")

        # 4. Sort by site and time if columns exist
        if site_col and time_col:
            df = df.sort_values(by=[site_col, time_col]).reset_index(drop=True)
            logger.info(f"Sorted data by {site_col} and {time_col}")

        # 5. Add derived features based on config
        if self.config.get('feature_engineering', {}).get('add_time_features', True):
            # Convert time column to datetime if it's not already
            if time_col and not pd.api.types.is_datetime64_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])

            # Add time-based features
            if time_col:
                df['hour_utc'] = df[time_col].dt.hour
                df['day_of_year'] = df[time_col].dt.dayofyear
                df['month'] = df[time_col].dt.month
                df['day_of_week'] = df[time_col].dt.dayofweek
                logger.info("Added time-based features")

        # 6. Add log-transformed variables for skewed distributions
        log_transform_vars = self.config.get('feature_engineering', {}).get('log_transform_variables', [])

        for var in log_transform_vars:
            if var in df.columns:
                # Add small offset to handle zeros
                offset = 0.1  # Default offset
                if isinstance(log_transform_vars, dict) and isinstance(log_transform_vars[var], dict):
                    offset = log_transform_vars[var].get('offset', 0.1)

                # Create log-transformed column
                df[f'log_{var}'] = np.log(df[var] + offset)
                logger.info(f"Added log-transformed feature for {var}")

        # 7. Calculate bias if both observation and model values are available
        target_var = self.config.get('training', {}).get('target_variable', 'PM25')
        model_target_var = f"model_{target_var}"

        if target_var in df.columns and model_target_var in df.columns:
            df['bias'] = df[target_var] - df[model_target_var]
            logger.info(f"Added bias column (observation - model) for {target_var}")

        logger.info(f"Data cleaning complete. Final dataset has {len(df)} rows and {len(df.columns)} columns")
        return df

    def split_train_test(self,
                        df: pd.DataFrame,
                        test_size: float = 0.2,
                        temporal_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and testing sets.

        Args:
            df: Cleaned DataFrame to split
            test_size: Fraction of data for testing
            temporal_split: Whether to split by time (True) or randomly (False)

        Returns:
            Tuple of (train_df, test_df)
        """
        split_config = self.config.get('training', {}).get('train_test_split', {})
        test_size = split_config.get('test_size', test_size)
        temporal_split = split_config.get('temporal_split', temporal_split)
        random_state = split_config.get('random_state', 42)

        if temporal_split:
            # Find time column
            time_col = None
            possible_time_cols = ['time_utc', 'time', 'datetime']
            for col in possible_time_cols:
                if col in df.columns:
                    time_col = col
                    break

            if time_col:
                # Sort by time
                df = df.sort_values(by=time_col)

                # Split at a time point
                split_idx = int(len(df) * (1 - test_size))
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()

                logger.info(f"Temporal split: training data from {df[time_col].iloc[0]} to {df[time_col].iloc[split_idx-1]}")
                logger.info(f"Temporal split: testing data from {df[time_col].iloc[split_idx]} to {df[time_col].iloc[-1]}")
            else:
                logger.warning("No time column found for temporal split, falling back to random split")
                train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        else:
            # Random split
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            logger.info(f"Random split: {len(train_df)} training samples, {len(test_df)} testing samples")

        return train_df, test_df

    def process_training_data(self,
                            start_date: Union[str, datetime],
                            end_date: Union[str, datetime],
                            save_output: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the full training data processing pipeline.

        Args:
            start_date: Start date for data processing
            end_date: End date for data processing
            save_output: Whether to save the processed data to disk

        Returns:
            Tuple of (full_df, train_df, test_df)
        """
        # 1. Get observation data
        obs_data = self.get_observation_data(
            start_date=start_date,
            end_date=end_date
        )

        # 2. Get model forecast data
        variables_to_process = self.config.get('model', {}).get('variables', None)
        categories_to_process = self.config.get('model', {}).get('categories', None)

        model_data = self.get_model_data(
            start_date=start_date,
            end_date=end_date,
            categories=categories_to_process,
            variables=variables_to_process
        )

        # 3. Interpolate model data to observation locations
        merged_df = self.interpolate_model_to_obs_locations(model_data, obs_data)

        # 4. Clean and prepare the merged data
        cleaned_df = self.clean_merged_data(merged_df)

        # 5. Split into training and testing sets
        train_df, test_df = self.split_train_test(cleaned_df)

        # 6. Save the processed data if requested
        if save_output:
            # Generate output file names
            date_str = f"{start_date.strftime('%Y%m%d') if isinstance(start_date, datetime) else start_date.replace('-', '')}_"
            date_str += f"{end_date.strftime('%Y%m%d') if isinstance(end_date, datetime) else end_date.replace('-', '')}"

            # Get target variable name
            target_var = self.config.get('training', {}).get('target_variable', 'PM25')

            # Save full dataset
            full_path = os.path.join(self.output_dir, f"{target_var}_{date_str}_FULL.csv")
            cleaned_df.to_csv(full_path, index=False)
            logger.info(f"Saved full dataset to {full_path}")

            # Save training set
            train_path = os.path.join(self.output_dir, f"{target_var}_{date_str}_TRAIN.csv")
            train_df.to_csv(train_path, index=False)
            logger.info(f"Saved training dataset to {train_path}")

            # Save testing set
            test_path = os.path.join(self.output_dir, f"{target_var}_{date_str}_TEST.csv")
            test_df.to_csv(test_path, index=False)
            logger.info(f"Saved testing dataset to {test_path}")

            # Create summary visualizations
            if self.config.get('output', {}).get('create_visualizations', True):
                self.create_visualizations(cleaned_df, train_df, test_df, target_var)

        return cleaned_df, train_df, test_df

    def create_visualizations(self,
                            full_df: pd.DataFrame,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            target_var: str) -> None:
        """
        Create visualizations of the processed data.

        Args:
            full_df: Complete processed DataFrame
            train_df: Training data DataFrame
            test_df: Testing data DataFrame
            target_var: Target variable name
        """
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Distribution of target variable
        plt.figure(figsize=(12, 6))
        plt.hist(full_df[target_var], bins=50, alpha=0.7, color='blue', label='Observed')

        model_target_var = f"model_{target_var}"
        if model_target_var in full_df.columns:
            plt.hist(full_df[model_target_var], bins=50, alpha=0.7, color='red', label='Model')
            plt.title(f'Distribution of Observed vs Model {target_var}')
            plt.legend()
        else:
            plt.title(f'Distribution of Observed {target_var}')

        plt.xlabel(target_var)
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, f'{target_var}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Scatter plot of observed vs model
        if model_target_var in full_df.columns:
            plt.figure(figsize=(10, 10))
            plt.scatter(full_df[target_var], full_df[model_target_var], alpha=0.3, s=10)

            # Add a 1:1 line
            max_val = max(full_df[target_var].max(), full_df[model_target_var].max())
            min_val = min(full_df[target_var].min(), full_df[model_target_var].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)

            plt.title(f'Observed vs Model {target_var}')
            plt.xlabel(f'Observed {target_var}')
            plt.ylabel(f'Model {target_var}')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.savefig(os.path.join(viz_dir, f'{target_var}_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Bias distribution
        if 'bias' in full_df.columns:
            plt.figure(figsize=(12, 6))
            plt.hist(full_df['bias'], bins=50, alpha=0.7, color='green')
            plt.axvline(0, color='red', linestyle='--')
            plt.title(f'Distribution of Bias (Observed - Model {target_var})')
            plt.xlabel('Bias')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, f'{target_var}_bias_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Time series of train/test split
        time_col = None
        possible_time_cols = ['time_utc', 'time', 'datetime']
        for col in possible_time_cols:
            if col in full_df.columns:
                time_col = col
                break

        if time_col:
            plt.figure(figsize=(14, 6))
            plt.scatter(train_df[time_col], train_df[target_var], alpha=0.3, s=3, color='blue', label='Train')
            plt.scatter(test_df[time_col], test_df[target_var], alpha=0.3, s=3, color='red', label='Test')
            plt.title(f'Time Series of {target_var} (Train/Test Split)')
            plt.xlabel('Time')
            plt.ylabel(target_var)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(viz_dir, f'{target_var}_train_test_timeseries.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Feature correlation heatmap
        numeric_cols = full_df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 1 and len(numeric_cols) <= 30:  # Only if we have a reasonable number of columns
            plt.figure(figsize=(14, 12))
            correlation = full_df[numeric_cols].corr()
            plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation')
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Created visualizations in {viz_dir}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Process observation and model data for AI training'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./config/training_data_config.yaml',
        help='Path to training data config file'
    )
    parser.add_argument(
        '--start-date', '-s',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', '-e',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Don't save output files"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for processed data'
    )

    args = parser.parse_args()

    # Initialize the processor
    processor = UnifiedTrainingDataProcessor(args.config)

    # Override output directory if specified
    if args.output_dir:
        processor.output_dir = Path(args.output_dir)
        processor.output_dir.mkdir(parents=True, exist_ok=True)

    # Get default dates from config if not specified
    if not args.start_date or not args.end_date:
        default_range = processor.config.get('processing', {}).get('default_time_range', {})
        if not args.start_date:
            args.start_date = default_range.get('start_date', '2025-01-01')
        if not args.end_date:
            args.end_date = default_range.get('end_date', '2025-04-30')

    # Process the data
    full_df, train_df, test_df = processor.process_training_data(
        start_date=args.start_date,
        end_date=args.end_date,
        save_output=not args.no_save
    )

    print(f"Data processing complete. Full dataset shape: {full_df.shape}")
    print(f"Training set: {len(train_df)} samples, Testing set: {len(test_df)} samples")


if __name__ == '__main__':
    main()
