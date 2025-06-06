#!/usr/bin/env python3
'''
DAFCOM Unified Observational Data Processor
Uses monetio for efficient retrieval and processing of air quality observation data

Author: GitHub Copilot
Date: June 5, 2025
'''

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging
import yaml
from typing import Dict, List, Optional, Union, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MONET I/O library for observation data access
try:
    import monet as mn
    import monetio as mio
    from monetio.obs import aeronet, airnow, aqs, cems, improve, tolnet, icartt
    MONETIO_AVAILABLE = True
except ImportError:
    logger.warning("monetio package not available. Install with: pip install git+https://github.com/noaa-oar-arl/monetio.git")
    MONETIO_AVAILABLE = False


class ObservationDataLoader:
    """
    Unified observation data loader using monetio package.
    Retrieves and processes air quality observations in a memory-efficient manner
    aligned with the DAFCOM unified processing approach.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the observation data loader.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Configure logging
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO').upper())
        logging.getLogger().setLevel(log_level)

        # Set output directories
        self.output_dir = Path(self.config.get('output', {}).get('output_dir', './processed_obs/'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ObservationDataLoader initialized with config from {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def get_airnow_data(self,
                      start_date: Union[str, datetime],
                      end_date: Union[str, datetime],
                      species: List[str] = ['PM25'],
                      bbox: List[float] = [25, -125, 49, -65],
                      resample: Optional[str] = 'H',
                      local_file: Optional[str] = None) -> xr.Dataset:
        """
        Get AirNow observation data either from local file or via API.

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            species: List of species to retrieve (PM25, OZONE, NO2, etc.)
            bbox: Bounding box [lat_min, lon_min, lat_max, lon_max]
            resample: Temporal resampling frequency (H for hourly, None for no resampling)
            local_file: Path to local NetCDF file (if None, will attempt download)

        Returns:
            xarray.Dataset with the processed observation data
        """
        if not MONETIO_AVAILABLE:
            raise ImportError("monetio package is required for observation processing")

        logger.info(f"Retrieving AirNow data for {species} from {start_date} to {end_date}")

        # Handle dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Option 1: Load from local file if specified
        if local_file:
            logger.info(f"Loading AirNow data from local file: {local_file}")
            try:
                # Use monetio's reader which handles the dataset structure properly
                df = mio.obs.airnow.load_airnow_file(local_file)

                # Convert to xarray for consistency with the rest of DAFCOM
                data = df.to_xarray()

            except Exception as e:
                logger.error(f"Error loading local AirNow file: {e}")
                raise

        # Option 2: Download from API
        else:
            logger.info(f"Downloading AirNow data from API")
            try:
                # Use monetio to download and process the data
                df = mio.obs.airnow.add_data(
                    start_date,
                    end_date,
                    param=species
                )

                # Convert to xarray for consistency
                data = df.to_xarray()

            except Exception as e:
                logger.error(f"Error downloading AirNow data: {e}")
                raise

        # Filter for bounding box
        if bbox:
            lat_min, lon_min, lat_max, lon_max = bbox
            logger.info(f"Filtering for bounding box: {bbox}")

            # Check if we have both 'latitude' and 'lat' columns (monetio format depends on source)
            lat_col = 'latitude' if 'latitude' in data.variables else 'lat'
            lon_col = 'longitude' if 'longitude' in data.variables else 'lon'

            mask = (
                (data[lat_col] >= lat_min) &
                (data[lat_col] <= lat_max) &
                (data[lon_col] >= lon_min) &
                (data[lon_col] <= lon_max)
            )

            data = data.where(mask, drop=True)

        # Resample to consistent temporal frequency if requested
        if resample:
            logger.info(f"Resampling to {resample} frequency")

            # Check which time dimension name is used
            time_dim = 'time' if 'time' in data.dims else 'datetime'

            # Group by station and resample
            # This is a simplification - proper resampling should be done by station
            data = data.resample({time_dim: resample}).mean()

        # Quality control - remove missing or invalid values
        for var in species:
            if var in data:
                # AirNow uses -1 as missing value flag
                var_data = data[var].where(data[var] > 0)
                data[var] = var_data

        logger.info(f"Retrieved {len(data.time) if 'time' in data.dims else len(data.datetime)} timestamps")

        return data

    def get_aqs_data(self,
                   start_date: Union[str, datetime],
                   end_date: Union[str, datetime],
                   species: List[str] = ['PM25'],
                   bbox: List[float] = [25, -125, 49, -65],
                   resample: Optional[str] = 'H') -> xr.Dataset:
        """
        Get EPA AQS observation data (similar structure to AirNow, but different source).

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            species: List of species to retrieve (PM25, OZONE, NO2, etc.)
            bbox: Bounding box [lat_min, lon_min, lat_max, lon_max]
            resample: Temporal resampling frequency (H for hourly, None for no resampling)

        Returns:
            xarray.Dataset with the processed observation data
        """
        if not MONETIO_AVAILABLE:
            raise ImportError("monetio package is required for observation processing")

        logger.info(f"Retrieving AQS data for {species} from {start_date} to {end_date}")

        # Handle dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        try:
            # Use monetio to access AQS data
            df = mio.obs.aqs.add_data(
                start_date,
                end_date,
                param=species
            )

            # Convert to xarray for consistency
            data = df.to_xarray()

        except Exception as e:
            logger.error(f"Error retrieving AQS data: {e}")
            raise

        # Apply same filtering and processing as AirNow
        # Filter for bounding box
        if bbox:
            lat_min, lon_min, lat_max, lon_max = bbox

            lat_col = 'latitude' if 'latitude' in data.variables else 'lat'
            lon_col = 'longitude' if 'longitude' in data.variables else 'lon'

            mask = (
                (data[lat_col] >= lat_min) &
                (data[lat_col] <= lat_max) &
                (data[lon_col] >= lon_min) &
                (data[lon_col] <= lon_max)
            )

            data = data.where(mask, drop=True)

        # Resample to consistent temporal frequency if requested
        if resample:
            time_dim = 'time' if 'time' in data.dims else 'datetime'
            data = data.resample({time_dim: resample}).mean()

        # Quality control
        for var in species:
            if var in data:
                # AQS uses negative values for flags
                var_data = data[var].where(data[var] >= 0)
                data[var] = var_data

        return data

    def save_processed_data(self,
                          data: xr.Dataset,
                          output_name: str,
                          format: str = 'netcdf') -> str:
        """
        Save processed observation data to file.

        Args:
            data: xarray Dataset with observation data
            output_name: Name for the output file
            format: Output format ('netcdf' or 'csv')

        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output path
        if format.lower() == 'netcdf':
            if not output_name.endswith('.nc'):
                output_name = f"{output_name}.nc"
            output_path = self.output_dir / output_name

            # Save as NetCDF
            logger.info(f"Saving data to NetCDF: {output_path}")

            # Get compression settings from config
            compression = self.config.get('output', {}).get('compression', {})
            encoding = {var: compression for var in data.data_vars}

            data.to_netcdf(output_path, encoding=encoding)

        elif format.lower() == 'csv':
            if not output_name.endswith('.csv'):
                output_name = f"{output_name}.csv"
            output_path = self.output_dir / output_name

            # Convert to DataFrame and save as CSV
            logger.info(f"Saving data to CSV: {output_path}")

            # Convert to pandas dataframe
            df = data.to_dataframe().reset_index()
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Data saved to {output_path}")
        return str(output_path)

    def combine_observation_sources(self, datasets: List[xr.Dataset],
                                  priority: Optional[List[str]] = None) -> xr.Dataset:
        """
        Combine observations from multiple sources with priority handling.

        Args:
            datasets: List of xarray Datasets to combine
            priority: List of source names in priority order (highest priority first)
                     If None, all sources are treated equally

        Returns:
            Combined xarray Dataset
        """
        if not datasets:
            logger.warning("No datasets provided to combine")
            return None

        if len(datasets) == 1:
            logger.info("Only one dataset provided, no combination needed")
            return datasets[0]

        # Merge datasets, handling conflicts according to priority
        # This is a simplified approach - in practice you'd want to handle
        # coordinates and metadata more carefully
        logger.info(f"Combining {len(datasets)} observation datasets")

        # Start with the highest priority dataset
        if priority:
            # Reorder datasets by priority
            ordered_datasets = []
            for source in priority:
                for ds in datasets:
                    if 'source' in ds.attrs and ds.attrs['source'] == source:
                        ordered_datasets.append(ds)

            # Add any remaining datasets
            for ds in datasets:
                if ds not in ordered_datasets:
                    ordered_datasets.append(ds)
        else:
            ordered_datasets = datasets

        # Combine datasets
        combined = ordered_datasets[0].copy()

        for ds in ordered_datasets[1:]:
            # For each variable in the new dataset
            for var in ds.data_vars:
                if var in combined:
                    # Where the current dataset has NaN and the new one doesn't, use the new value
                    combined[var] = xr.where(
                        combined[var].isnull() & ~ds[var].isnull(),
                        ds[var],
                        combined[var]
                    )
                else:
                    # If the variable doesn't exist in the combined dataset, add it
                    combined[var] = ds[var]

        return combined

    def process_observations(self,
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime],
                           species: List[str] = ['PM25'],
                           sources: List[str] = ['airnow'],
                           bbox: List[float] = [25, -125, 49, -65],
                           resample: str = 'H',
                           output_name: Optional[str] = None,
                           output_format: str = 'netcdf',
                           local_files: Optional[Dict[str, str]] = None) -> xr.Dataset:
        """
        Process observation data from multiple sources.

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            species: List of species to retrieve
            sources: List of sources to use ('airnow', 'aqs', 'improve', etc.)
            bbox: Bounding box [lat_min, lon_min, lat_max, lon_max]
            resample: Temporal resampling frequency
            output_name: Name for the output file (if None, will generate based on dates)
            output_format: Output format ('netcdf' or 'csv')
            local_files: Dictionary mapping source names to local file paths

        Returns:
            Combined xarray Dataset with all processed observations
        """
        if not MONETIO_AVAILABLE:
            raise ImportError("monetio package is required for observation processing")

        # Handle dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Generate default output name if not provided
        if output_name is None:
            date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            species_str = '_'.join(species)
            output_name = f"obs_{species_str}_{date_str}"

        # Process each source
        datasets = []

        for source in sources:
            logger.info(f"Processing {source} data")

            local_file = None
            if local_files and source in local_files:
                local_file = local_files[source]

            if source.lower() == 'airnow':
                data = self.get_airnow_data(
                    start_date=start_date,
                    end_date=end_date,
                    species=species,
                    bbox=bbox,
                    resample=resample,
                    local_file=local_file
                )

            elif source.lower() == 'aqs':
                data = self.get_aqs_data(
                    start_date=start_date,
                    end_date=end_date,
                    species=species,
                    bbox=bbox,
                    resample=resample
                )

            # Add other sources as needed...

            else:
                logger.warning(f"Unsupported source: {source}")
                continue

            # Add source metadata
            data.attrs['source'] = source

            datasets.append(data)

        # Combine all datasets
        # Priority order: AirNow > AQS > others
        combined = self.combine_observation_sources(
            datasets,
            priority=['airnow', 'aqs', 'improve']
        )

        # Save the combined dataset
        if combined is not None:
            self.save_processed_data(
                combined,
                output_name=output_name,
                format=output_format
            )

        return combined


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Process observational data for DAFCOM using monetio'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='../config/obs_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--species', '-s',
        type=str,
        nargs='+',
        default=['PM25'],
        help='Species to retrieve (e.g., PM25, OZONE, NO2)'
    )
    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        default=['airnow'],
        help='Data sources to use (e.g., airnow, aqs)'
    )
    parser.add_argument(
        '--local-file',
        type=str,
        help='Path to local data file (if using local data)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        help='Output file name'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='netcdf',
        choices=['netcdf', 'csv'],
        help='Output format'
    )

    args = parser.parse_args()

    # Load configuration first, then override with command line arguments
    loader = ObservationDataLoader(args.config)

    # Default date range if not specified
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    # Process data
    local_files = {args.sources[0]: args.local_file} if args.local_file else None

    processed = loader.process_observations(
        start_date=args.start_date,
        end_date=args.end_date,
        species=args.species,
        sources=args.sources,
        output_name=args.output_name,
        output_format=args.output_format,
        local_files=local_files
    )

    print(f"Processed {len(processed.time) if 'time' in processed.dims else 0} timepoints for {args.species}")


if __name__ == '__main__':
    main()
