"""
This file contains mock classes to be used during Sphinx documentation generation.
These mock classes serve as stand-ins for actual modules that might be difficult
to import during the documentation build process.
"""

class UnifiedTrainingDataProcessor:
    """
    Mock class for UnifiedTrainingDataProcessor to be used in Sphinx documentation.
    """
    def __init__(self, config_path):
        """
        Initialize the processor with a configuration file.

        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path

    def get_observation_data(self, start_date, end_date, species=None, bbox=None):
        """
        Retrieve and process observation data

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            species: List of species to retrieve (default from config)
            bbox: Bounding box [lat_min, lon_min, lat_max, lon_max]

        Returns:
            xarray.Dataset with processed observation data
        """
        pass

    def get_model_data(self, start_date, end_date, categories=None, variables=None):
        """
        Retrieve and process model forecast data

        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            categories: List of categories (meteorology, chemistry, aerosol)
            variables: List of specific variables to retrieve

        Returns:
            xarray.Dataset with processed model data
        """
        pass

    def interpolate_model_to_obs_locations(self, model_data, obs_data):
        """
        Interpolate model data to observation locations

        Args:
            model_data: Model forecast data as xarray Dataset
            obs_data: Observation data as xarray Dataset

        Returns:
            DataFrame with model values at observation locations
        """
        pass

class RegriddingProcessor:
    """
    Handles spatial regridding of data between different coordinate systems and resolutions.
    """
    pass

class ParallelDataLoader:
    """
    Loads atmospheric data in parallel for improved performance.
    """
    pass

class ObservationDataLoader:
    """
    Specialized loader for air quality observations from multiple sources.
    """
    pass
