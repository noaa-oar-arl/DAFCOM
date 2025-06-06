#!/usr/bin/env python3
"""
DAFCOM Observation Data Processing Example

This example demonstrates how to use the new monetio-based unified observation
processor to retrieve and process air quality observations efficiently.

Author: GitHub Copilot
Date: June 5, 2025
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from unified_obs_processor import ObservationDataLoader


def process_airnow_example(config_path, start_date=None, end_date=None, local_file=None, plot=True):
    """
    Process AirNow data using the unified approach with monetio.

    Args:
        config_path: Path to configuration file
        start_date: Start date (if None, use 30 days ago)
        end_date: End date (if None, use today)
        local_file: Path to local AirNow file (if None, try to download)
        plot: Whether to create plots
    """
    print("\n===== Processing AirNow Data with MONET I/O =====")

    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Date range: {start_date} to {end_date}")

    # Initialize the observation data loader
    loader = ObservationDataLoader(config_path)

    # Process the data
    local_files = {'airnow': local_file} if local_file else None

    obs_data = loader.process_observations(
        start_date=start_date,
        end_date=end_date,
        species=['PM25'],
        sources=['airnow'],
        output_name=f"airnow_pm25_{start_date.replace('-', '')}_{end_date.replace('-', '')}",
        output_format='netcdf',
        local_files=local_files
    )

    # Print summary statistics
    if obs_data is not None:
        print("\n=== Data Summary ===")
        print(f"Time range: {obs_data.time.min().values} to {obs_data.time.max().values}")
        print(f"Number of time steps: {len(obs_data.time)}")

        # Get latitude/longitude variable names (depends on source format)
        lat_var = 'latitude' if 'latitude' in obs_data.variables else 'lat'
        lon_var = 'longitude' if 'longitude' in obs_data.variables else 'lon'

        print(f"Number of stations: {len(obs_data[lat_var])}")
        print(f"Latitude range: {obs_data[lat_var].min().values} to {obs_data[lat_var].max().values}")
        print(f"Longitude range: {obs_data[lon_var].min().values} to {obs_data[lon_var].max().values}")

        if 'PM25' in obs_data:
            print("\n=== PM2.5 Statistics ===")
            pm_values = obs_data['PM25'].values
            valid_values = pm_values[~np.isnan(pm_values)]
            print(f"Valid measurements: {len(valid_values)}")
            print(f"Mean: {np.mean(valid_values):.2f} µg/m³")
            print(f"Median: {np.median(valid_values):.2f} µg/m³")
            print(f"Min: {np.min(valid_values):.2f} µg/m³")
            print(f"Max: {np.max(valid_values):.2f} µg/m³")

        # Create plots if requested
        if plot:
            create_plots(obs_data, start_date, end_date)

    else:
        print("No data was returned. Check logs for errors.")

    return obs_data


def create_plots(obs_data, start_date, end_date):
    """Create example plots from the observation data."""
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)

    # Determine variable names based on what's in the dataset
    lat_var = 'latitude' if 'latitude' in obs_data.variables else 'lat'
    lon_var = 'longitude' if 'longitude' in obs_data.variables else 'lon'

    # Plot 1: Map of station locations
    plt.figure(figsize=(12, 8))
    plt.scatter(obs_data[lon_var], obs_data[lat_var], c='blue', alpha=0.7, s=10)
    plt.title(f'AirNow Monitoring Stations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-125, -65)
    plt.ylim(25, 50)

    # Add US state boundaries if basemap or cartopy is available
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES)
        ax.scatter(obs_data[lon_var], obs_data[lat_var], c='blue', alpha=0.7, s=10,
                  transform=ccrs.PlateCarree())
        ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())
        plt.title(f'AirNow Monitoring Stations')
        plt.savefig(f"{output_dir}/station_map_cartopy.png", dpi=300, bbox_inches='tight')
    except ImportError:
        pass  # Skip if cartopy isn't installed

    plt.savefig(f"{output_dir}/station_map.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Time series of average PM2.5
    if 'PM25' in obs_data:
        # Convert to pandas for easier time series manipulation
        if 'site' in obs_data.dims:  # Some monetio formats have a site dimension
            # Average across all sites for each timestamp
            pm_avg = obs_data['PM25'].mean(dim='site').to_dataframe()
        else:
            # First convert to DataFrame
            pm_df = obs_data['PM25'].to_dataframe()
            # Then group by time and average
            pm_avg = pm_df.groupby('time').mean()

        plt.figure(figsize=(14, 6))
        pm_avg.plot(figsize=(14, 6), color='red', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Average PM2.5 Concentration ({start_date} to {end_date})')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pm25_timeseries.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 3: Histogram of PM2.5 values
    if 'PM25' in obs_data:
        plt.figure(figsize=(10, 6))
        pm_values = obs_data['PM25'].values
        valid_values = pm_values[~np.isnan(pm_values)]

        plt.hist(valid_values, bins=30, alpha=0.7, color='green')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Distribution of PM2.5 Measurements')
        plt.xlabel('PM2.5 (µg/m³)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pm25_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}/")


def compare_with_old_approach(old_file, new_data):
    """Compare the new approach with the old Excel-based approach."""
    print("\n===== Comparing New Approach with Old Approach =====")

    if not os.path.exists(old_file):
        print(f"Old file not found: {old_file}")
        return

    # Load the old Excel file
    try:
        old_data = pd.read_excel(old_file)
        print(f"Old data loaded from {old_file}")
        print(f"Old data shape: {old_data.shape}")

        # Basic statistics from old data
        print("\n=== Old Data Statistics ===")
        print(f"Number of records: {len(old_data)}")
        print(f"PM2.5 mean: {old_data['airnow_pm25'].mean():.2f}")
        print(f"PM2.5 median: {old_data['airnow_pm25'].median():.2f}")
        print(f"PM2.5 min: {old_data['airnow_pm25'].min():.2f}")
        print(f"PM2.5 max: {old_data['airnow_pm25'].max():.2f}")

        # Compare with new data
        lat_var = 'latitude' if 'latitude' in new_data.variables else 'lat'
        lon_var = 'longitude' if 'longitude' in new_data.variables else 'lon'

        # Convert new xarray data to pandas for comparison
        if 'PM25' in new_data:
            new_df = new_data['PM25'].to_dataframe().reset_index()

            print("\n=== Comparison ===")
            print(f"Old records: {len(old_data)}")
            print(f"New records: {len(new_df)}")

            # Memory usage comparison
            old_memory = old_data.memory_usage(deep=True).sum() / 1e6
            new_memory = new_df.memory_usage(deep=True).sum() / 1e6
            print(f"Old data memory usage: {old_memory:.2f} MB")
            print(f"New data memory usage: {new_memory:.2f} MB")
            print(f"Memory improvement: {(1 - new_memory/old_memory) * 100:.2f}%")

            # Save comparison to CSV
            comparison = pd.DataFrame({
                'Metric': ['Records', 'Mean PM2.5', 'Median PM2.5', 'Min PM2.5', 'Max PM2.5', 'Memory (MB)'],
                'Old Approach': [len(old_data), old_data['airnow_pm25'].mean(),
                               old_data['airnow_pm25'].median(),
                               old_data['airnow_pm25'].min(),
                               old_data['airnow_pm25'].max(),
                               old_memory],
                'New Approach': [len(new_df), new_df['PM25'].mean(),
                               new_df['PM25'].median(),
                               new_df['PM25'].min(),
                               new_df['PM25'].max(),
                               new_memory],
            })

            comparison.to_csv("approach_comparison.csv", index=False)
            print("\nComparison saved to approach_comparison.csv")

    except Exception as e:
        print(f"Error comparing approaches: {e}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='DAFCOM Observation Data Processing Example')
    parser.add_argument('--config', '-c', type=str, default='./config/obs_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--start-date', '-s', type=str,
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e', type=str,
                      help='End date (YYYY-MM-DD)')
    parser.add_argument('--local-file', '-l', type=str,
                      help='Path to local AirNow file')
    parser.add_argument('--compare-old', '-o', type=str,
                      default='./getobs_PM25_20250401_20250501.xlsx',
                      help='Path to old Excel file for comparison')
    parser.add_argument('--no-plot', action='store_true',
                      help='Skip creating plots')

    args = parser.parse_args()

    # Process data with new approach
    new_data = process_airnow_example(
        args.config,
        args.start_date,
        args.end_date,
        args.local_file,
        not args.no_plot
    )

    # Compare with old approach
    if args.compare_old and os.path.exists(args.compare_old) and new_data is not None:
        compare_with_old_approach(args.compare_old, new_data)


if __name__ == '__main__':
    main()
