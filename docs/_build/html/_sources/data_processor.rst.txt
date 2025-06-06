=======================
DAFCOM Data Processor
=======================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

The DAFCOM Data Processor is a core component of the Data Analytics for Air Quality Forecasting with Operational Models
framework. It handles the preprocessing, transformation, and preparation of various input data sources required for both
training machine learning models and generating operational air quality forecasts.

The data processor operates in two primary modes:

1. **Reanalysis Mode**: Processes historical data for training and validation of machine learning models
2. **Forecast Mode**: Processes real-time data for operational air quality predictions

Key Capabilities
-----------------

* **Data Ingestion**: Reads various formats of meteorological, air quality, and satellite data
* **Spatial Processing**: Performs regridding, interpolation, and coordinate transformations
* **Temporal Processing**: Handles time series alignment, aggregation, and feature engineering
* **Data Cleaning**: Performs quality control, gap filling, and outlier detection
* **Data Fusion**: Combines multiple data sources into unified datasets
* **Feature Engineering**: Creates derived variables and transforms for machine learning
* **Data Export**: Saves processed data in various formats for downstream analysis

Architecture
============

.. note::

   The DAFCOM Data Processor architecture includes several key components:

   * **Input Data Sources**: Meteorological data, satellite observations, air quality measurements
   * **Processing Components**: File I/O, spatial/temporal processing, quality control, data fusion
   * **Output Products**: Training datasets, validation datasets, forecast inputs
   * **Processing Modes**: Reanalysis (historical) and Forecast (operational) modes

The data processor is organized into a collection of modular components that can be combined into processing pipelines:

* **File I/O Modules**: Handle reading and writing data in various formats (NetCDF, CSV, HDF5)
* **Spatial Processors**: Perform regridding, interpolation, and coordinate transformations
* **Temporal Processors**: Handle time series alignment, resampling, and time-based feature engineering
* **Quality Control Modules**: Handle missing data, outliers, and data validation
* **Fusion Modules**: Combine multiple data sources using various techniques
* **Feature Engineering Modules**: Create derived variables for machine learning

Data Processing Workflows
==========================

Reanalysis Mode
----------------

In reanalysis mode, the data processor prepares historical data for training machine learning models. The workflow typically includes:

1. **Historical Data Collection**: Gather meteorological, air quality, and satellite data for the training period
2. **Consistent Grid Creation**: Regrid all input data to a common spatial resolution
3. **Temporal Alignment**: Ensure all data sources are temporally aligned
4. **Feature Engineering**: Create additional features like temporal aggregations, spatial statistics, and derived variables
5. **Quality Control**: Filter out bad data points and fill gaps where possible
6. **Train/Test Splitting**: Prepare data for model training and validation
7. **Export**: Save processed data in ML-ready formats

The reanalysis workflow processes multiple data types:

* **Meteorological Data**: Temperature, wind, boundary layer height, etc.
* **Satellite Data**: AOD, NDVI, land use/land cover, etc.
* **Air Quality Observations**: PM2.5, NO2, Ozone, etc.
* **Emissions Data**: Anthropogenic and biogenic emissions
* **Static Features**: Population density, road networks, elevation, etc.

Forecast Mode
--------------

In forecast mode, the data processor prepares real-time data for operational air quality predictions:

1. **Real-time Data Collection**: Gather latest meteorological forecasts, satellite data, and air quality observations
2. **Consistent Grid Creation**: Regrid all input data to a common spatial resolution
3. **Feature Engineering**: Apply the same transformations used during model training
4. **Missing Data Handling**: Apply strategies to handle data gaps in real-time context
5. **Export**: Prepare data for inference by trained models

Configuration
=============

The data processor is highly configurable through YAML configuration files. Configuration parameters include:

* **Input/Output Paths**: Locations for input data and output directories
* **Spatial Settings**: Grid resolutions, domains, and interpolation methods
* **Temporal Settings**: Time ranges, frequencies, and aggregation methods
* **Processing Options**: Quality control thresholds, feature engineering options
* **Logging Options**: Verbosity and log file locations

Example Configuration
---------------------

Below is a simplified example of the data processor configuration file. Complete example configuration files can be found in the ``config`` directory:

- ``data_processor_config.yaml`` - Comprehensive example with all available options
- ``reanalysis_mode_config.yaml`` - Configuration specific to reanalysis mode
- ``forecast_mode_config.yaml`` - Configuration specific to forecast mode

.. code-block:: yaml

    # Data processor configuration
    name: "DAFCOM Data Processor Configuration"
    version: "0.1.0"
    mode: "reanalysis"  # Options: reanalysis, forecast

    logging:
      level: "INFO"
      file: "/path/to/logs/data_processor.log"
      console_output: true

    paths:
      input:
        base_dir: "/path/to/input/data"
        meteorology: "${paths.input.base_dir}/meteorology"
        satellite:
          aod: "${paths.input.base_dir}/satellite/aod"
        observations: "${paths.input.base_dir}/observations"

      output:
        base_dir: "/path/to/output/data"
        processed: "${paths.output.base_dir}/processed"
        visualizations: "${paths.output.base_dir}/visualizations"

    grid:
      resolution: 0.01  # degrees
      bbox:
        lat_min: 25.0
        lon_min: -125.0
        lat_max: 49.0
        lon_max: -65.0
      interpolation:
        method: "bilinear"

    time:
      start_date: "2025-01-01"
      end_date: "2025-06-01"
      frequency: "1H"  # 1 hour

    processing:
      fill_gaps:
        enabled: true
        method: "interpolate"
      quality_control:
        remove_outliers: true
        z_score_threshold: 3.0
      feature_engineering:
        create_time_features: true
        transformations:
          - name: "log"
            variables: ["PM25", "NO2"]

File I/O Operations
====================

The data processor supports multiple file formats and data sources:

Input Formats
--------------

* **NetCDF**: Meteorological data, model outputs, gridded observations
* **CSV**: Point observations, station metadata
* **HDF5**: Satellite products, large arrays
* **GeoTIFF**: Land use data, static features
* **Shapefile**: Administrative boundaries, road networks

Output Formats
---------------

* **NetCDF**: Processed gridded data
* **CSV/Parquet**: ML-ready tabular data
* **Zarr**: Cloud-optimized data for distributed processing
* **HDF5**: Hierarchical data format for complex datasets

The data processor handles different time dimensions, coordinate systems, and metadata structures, normalizing them into consistent formats for downstream processing.

Regridding and Interpolation
=============================

The data processor uses multiple regridding and interpolation methods to transform data between different spatial resolutions and projections:

Regridding Methods
------------------

* **Bilinear**: Standard bilinear interpolation for continuous fields
* **Nearest Neighbor**: For categorical data like land use/land cover
* **Conservative**: For flux-preserving transformations
* **Patch Recovery**: For better handling of discontinuities

Vertical Interpolation
----------------------

For 3D meteorological fields, the processor can interpolate between pressure levels or height levels using:

* **Linear**: Standard pressure/height interpolation
* **Log-Linear**: For pressure level interpolation
* **Cubic Spline**: For smoother interpolation

Example Regridding Workflow
---------------------------

1. Define source and target grids
2. Select appropriate regridding method
3. Apply regridding with careful handling of coordinate systems
4. Validate results with quality checks
5. Add appropriate metadata to track the transformation

AOD Data Processing
===================

Aerosol Optical Depth (AOD) data requires special processing due to gaps in satellite retrievals. The DAFCOM data processor includes specialized workflows for AOD:

AOD Gap Filling
---------------

1. **Multi-Satellite Fusion**: Combine MODIS Terra, MODIS Aqua, and VIIRS data
2. **Model-Based Gap Filling**: Use model AOD to fill gaps in satellite retrievals
3. **Spatial Interpolation**: Apply smart interpolation methods that respect spatial patterns
4. **Temporal Interpolation**: Fill small temporal gaps using time series methods

AOD Quality Control
-------------------

* Filter AOD based on quality flags
* Remove cloud-contaminated pixels
* Apply consistency checks across multiple retrievals
* Handle differences in satellite overpass times

Meteorological Data Processing
==============================

The processor handles various meteorological variables essential for air quality prediction:

Key Meteorological Variables
----------------------------

* **2m Temperature**: Surface air temperature
* **10m Wind**: Wind speed and direction near the surface
* **Boundary Layer Height**: Height of the atmospheric boundary layer
* **Relative Humidity**: Atmospheric moisture content
* **Precipitation**: Rainfall amounts
* **Surface Pressure**: Atmospheric pressure at the surface
* **Radiation**: Solar radiation reaching the surface

Processing Steps
----------------

1. **Regridding**: Transform NWP model output to the common DAFCOM grid
2. **Temporal Interpolation**: Ensure consistent temporal frequency
3. **Derived Variables**: Calculate additional variables like ventilation coefficient
4. **Quality Control**: Check for physical consistency and unrealistic values

Air Quality Data Processing
===========================

The processor handles observations from monitoring networks and prepares them for model training:

Data Sources
------------

* **AirNow**: Real-time air quality monitoring network
* **AQS**: EPA's Air Quality System
* **CAMS**: Copernicus Atmosphere Monitoring Service
* **Custom Networks**: Project-specific monitoring data

Processing Steps
----------------

1. **Data Cleaning**: Remove invalid or flagged observations
2. **Unit Conversion**: Standardize units across different networks
3. **Temporal Aggregation**: Create hourly, daily, or other temporal averages
4. **Spatial Integration**: Match observations to the common grid
5. **Metadata Enhancement**: Add site information, measurement methods, etc.

Feature Engineering
===================

The data processor creates derived features to enhance model performance:

Temporal Features
---------------

* **Time of Day**: Hour, part of day
* **Day of Week**: Weekday/weekend patterns
* **Season**: Seasonal patterns
* **Holiday Flags**: Special days with different emissions patterns
* **Lag Features**: Previous observations for time series modeling
* **Moving Averages**: Rolling statistics over various windows

Spatial Features
--------------

* **Distance to Sources**: Proximity to roads, industrial sites
* **Land Use Proportions**: Percentage of different land use types in surrounding areas
* **Population Density**: Local and regional population metrics
* **Elevation**: Absolute elevation and relative terrain features
* **Spatial Lag**: Values at neighboring grid cells

Mathematical Transformations
-------------------------

* **Log Transformations**: For skewed variables like PM2.5
* **Normalization**: Standardizing numeric features
* **Dimensionality Reduction**: PCA or other techniques for highly correlated variables
* **Polynomial Features**: For capturing non-linear relationships

API Reference
============

The data processor exposes several key Python classes for programmatic use:

UnifiedTrainingDataProcessor
--------------------------

Main class for preparing training data that combines observations and model forecasts.

.. py:class:: UnifiedTrainingDataProcessor(config_path)

   :param config_path: Path to configuration YAML file
   :type config_path: str or Path

   .. py:method:: get_observation_data(start_date, end_date, species=None, bbox=None)

      Retrieve and process observation data

      :param start_date: Start date for retrieval
      :param end_date: End date for retrieval
      :param species: List of species to retrieve (default from config)
      :param bbox: Bounding box [lat_min, lon_min, lat_max, lon_max]
      :return: xarray.Dataset with processed observation data

   .. py:method:: get_model_data(start_date, end_date, categories=None, variables=None)

      Retrieve and process model forecast data

      :param start_date: Start date for retrieval
      :param end_date: End date for retrieval
      :param categories: List of categories (meteorology, chemistry, aerosol)
      :param variables: List of specific variables to retrieve
      :return: xarray.Dataset with processed model data

   .. py:method:: interpolate_model_to_obs_locations(model_data, obs_data)

      Interpolate model data to observation locations

      :param model_data: Model forecast data as xarray Dataset
      :param obs_data: Observation data as xarray Dataset
      :return: DataFrame with model values at observation locations

RegriddingProcessor
-----------------

Handles spatial regridding of data between different coordinate systems and resolutions.

ParallelDataLoader
----------------

Loads atmospheric data in parallel for improved performance.

ObservationDataLoader
-------------------

Specialized loader for air quality observations from multiple sources.

Command Line Interface
====================

The data processor can be run directly from the command line. The configuration file is the primary way to control behavior - use the examples from the ``config`` directory as templates for your own processing needs.

.. code-block:: bash

    # Run the unified training data processor with a general configuration
    python unified_training_processor.py --config config/data_processor_config.yaml --start-date 2025-01-01 --end-date 2025-05-31

    # Run with a mode-specific configuration
    python unified_training_processor.py --config config/reanalysis_mode_config.yaml --start-date 2025-01-01 --end-date 2025-05-31

    # Run a specific regridding task
    python step3_regrid_0p01.py --variable pm25 --input input.nc --output pm25_0p01.nc

    # Process AOD data with gap filling
    python process_aod_workflow.py --config config/data_processor_config.yaml --fill-gaps

Examples
========

Example 1: Processing Meteorological Data
---------------------------------------

.. code-block:: python

    from dafcom.processor import MeteoDataProcessor

    # Initialize processor with configuration
    processor = MeteoDataProcessor('meteo_config.yaml')

    # Process temperature data
    temp_data = processor.process_variable(
        variable='t2m',
        start_date='2025-01-01',
        end_date='2025-01-31'
    )

    # Save processed data
    processor.save_data(temp_data, 'processed_temperature.nc')

Example 2: Combining Observations and Model Data
----------------------------------------------

.. code-block:: python

    from dafcom.processor import UnifiedTrainingDataProcessor

    # Initialize the processor
    processor = UnifiedTrainingDataProcessor('training_config.yaml')

    # Get observation data
    obs_data = processor.get_observation_data(
        start_date='2025-01-01',
        end_date='2025-06-01',
        species=['PM25']
    )

    # Get model data
    model_data = processor.get_model_data(
        start_date='2025-01-01',
        end_date='2025-06-01',
        categories=['meteorology', 'chemistry']
    )

    # Combine data sources
    combined_data = processor.interpolate_model_to_obs_locations(
        model_data=model_data,
        obs_data=obs_data
    )

    # Perform quality control and feature engineering
    training_data = processor.prepare_training_data(combined_data)

    # Split and save training data
    processor.split_and_save_data(training_data, test_size=0.2)

Troubleshooting
==============

Common Issues
-----------

1. **Missing Data Errors**: Check input data availability and quality flags
2. **Memory Errors**: Try chunking data or reducing domain size for large datasets
3. **Regridding Artifacts**: Verify grid definitions and try alternative methods
4. **Coordinate System Mismatches**: Ensure consistent projection definitions
5. **Temporal Alignment Issues**: Check for timezone differences or metadata inconsistencies

Diagnostic Tools
--------------

The data processor includes several diagnostic tools:

* **Data Validation**: Scripts to check data validity before processing
* **Visualization Tools**: Plot intermediate results to verify processing steps
* **Performance Profiling**: Identify bottlenecks in processing pipelines
* **Log Analysis**: Extract structured information from detailed logs

Best Practices
=============

1. **Validate Input Data**: Always check data quality before processing
2. **Test on Subsets**: Test processing steps on small spatial/temporal subsets first
3. **Benchmark Results**: Compare processed data against expected values
4. **Document Transformations**: Keep track of all processing steps applied
5. **Version Control Configs**: Maintain version control for processing configurations
6. **Archive Raw Inputs**: Always keep original data to enable reprocessing
7. **Modular Processing**: Break complex workflows into smaller, testable components
8. **Consistent Metadata**: Ensure metadata is preserved throughout processing

Performance Optimization
=====================

The data processor employs several strategies to optimize performance:

1. **Parallel Processing**: Distributes work across multiple cores
2. **Chunked I/O**: Processes data in manageable chunks to reduce memory usage
3. **Lazy Evaluation**: Uses dask for delayed computation when appropriate
4. **Cache Management**: Intelligent caching of intermediate results
5. **Selective Processing**: Only processes required variables and domains
6. **Algorithm Selection**: Uses optimized implementations of costly operations
7. **Memory Mapping**: Accesses large datasets without loading them entirely into memory

Conclusion
=========

The DAFCOM Data Processor provides a robust foundation for preparing air quality data for both research and operational applications. Its modular design, extensive configuration options, and specialized processing workflows enable efficient handling of diverse data sources to create high-quality inputs for machine learning and forecasting applications.

For additional details, see the :doc:`API documentation <api/data_processor>` and :doc:`example notebooks <examples>`.

For complete configuration examples, refer to the following files in the project's ``config`` directory:

* ``data_processor_config.yaml`` - Comprehensive configuration with all available options
* ``reanalysis_mode_config.yaml`` - Specific configuration for historical data processing
* ``forecast_mode_config.yaml`` - Specific configuration for operational forecasting
