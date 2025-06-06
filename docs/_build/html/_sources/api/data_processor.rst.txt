Data Processor API Reference
==========================

This document contains API reference for key classes in the DAFCOM data processor.
Due to the project structure, some classes may not be directly importable by Sphinx.

UnifiedTrainingDataProcessor
---------------------------

.. py:class:: UnifiedTrainingDataProcessor(config_path)
   :no-index:

   Main class for preparing training data that combines observations and model forecasts.

   :param config_path: Path to configuration YAML file
   :type config_path: str or Path

   .. py:method:: get_observation_data(start_date, end_date, species=None, bbox=None)
      :no-index:

      Retrieve and process observation data

      :param start_date: Start date for retrieval
      :param end_date: End date for retrieval
      :param species: List of species to retrieve (default from config)
      :param bbox: Bounding box [lat_min, lon_min, lat_max, lon_max]
      :return: xarray.Dataset with processed observation data

   .. py:method:: get_model_data(start_date, end_date, categories=None, variables=None)
      :no-index:

      Retrieve and process model forecast data

      :param start_date: Start date for retrieval
      :param end_date: End date for retrieval
      :param categories: List of categories (meteorology, chemistry, aerosol)
      :param variables: List of specific variables to retrieve
      :return: xarray.Dataset with processed model data

   .. py:method:: interpolate_model_to_obs_locations(model_data, obs_data)
      :no-index:

      Interpolate model data to observation locations

      :param model_data: Model forecast data as xarray Dataset
      :param obs_data: Observation data as xarray Dataset
      :return: DataFrame with model values at observation locations

RegriddingProcessor
-----------------

.. py:class:: RegriddingProcessor
   :no-index:

   Handles spatial regridding of data between different coordinate systems and resolutions.

ParallelDataLoader
----------------

.. py:class:: ParallelDataLoader
   :no-index:

   Loads atmospheric data in parallel for improved performance.

ObservationDataLoader
-------------------

.. py:class:: ObservationDataLoader
   :no-index:

   Specialized loader for air quality observations from multiple sources.
