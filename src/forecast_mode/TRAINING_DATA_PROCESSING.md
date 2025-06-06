# Unified Training Data Processing for DAFCOM

This document describes the unified training data processor component of DAFCOM (Data Analytics for Air Quality Forecasting with Operational Models), which combines observation data with model forecast data to create ML-ready datasets for training air quality prediction models.

## Overview

The unified training data processor brings together all the previous components in the DAFCOM workflow:
1. It uses the unified atmospheric data processor to retrieve model forecast data
2. It uses the monetio-based observation processor to retrieve observation data
3. It aligns these datasets spatially and temporally
4. It applies quality control and feature engineering
5. It splits data into training and testing sets
6. It creates visualizations and saves data in ML-ready formats

![DAFCOM Workflow](../docs/images/training_workflow.png)

## Key Features

- **Unified Data Processing**: Combines observation and model data into a single workflow
- **Spatial Interpolation**: Interpolates model data to observation locations
- **Quality Control**: Applies configurable data cleaning and filtering
- **Feature Engineering**: Creates derived features (time features, log transforms)
- **Train/Test Splitting**: Supports both temporal and random splitting methods
- **Visualization**: Generates diagnostic plots for data exploration
- **Configurable Pipeline**: All aspects controlled via YAML configuration

## Installation

The unified training processor is part of the DAFCOM package. Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

Specific dependencies for the training processor include:
- pandas
- numpy
- xarray
- matplotlib
- scikit-learn
- dask (optional, for parallel processing)

## Configuration

The unified training processor uses a YAML configuration file that specifies:

1. **Data Sources**: Where to find observation and model data
2. **Processing Parameters**: How to process and clean the data
3. **Feature Engineering**: What derived features to create
4. **Output Settings**: Where and how to save the results

Example configuration:

```yaml
# Basic configuration structure
logging:
  level: INFO

paths:
  obs_config: "/path/to/obs_config.yaml"
  model_config: "/path/to/model_config.yaml"

processing:
  default_time_range:
    start_date: "2025-04-20"
    end_date: "2025-05-01"

observations:
  species: ["PM25"]
  sources: ["airnow", "aqs"]

model:
  categories: ["chemistry", "meteorology"]

# ... more settings ...
```

See the full [training_data_config.yaml](../config/training_data_config.yaml) for all available options.

## Usage

### Command Line

The processor can be run from the command line:

```bash
python unified_training_processor.py --config /path/to/config.yaml --start-date 2025-04-01 --end-date 2025-05-01
```

Command line arguments:
- `--config` or `-c`: Path to configuration file
- `--start-date` or `-s`: Start date (YYYY-MM-DD)
- `--end-date` or `-e`: End date (YYYY-MM-DD)
- `--output-dir` or `-o`: Output directory
- `--no-save`: Don't save output files (for testing)

### Python API

You can also use the processor programmatically:

```python
from unified_training_processor import UnifiedTrainingDataProcessor

# Initialize with configuration
processor = UnifiedTrainingDataProcessor('/path/to/config.yaml')

# Process all data at once
full_df, train_df, test_df = processor.process_training_data(
    start_date="2025-04-01",
    end_date="2025-05-01",
    save_output=True
)

# Or process step by step
obs_data = processor.get_observation_data(start_date, end_date)
model_data = processor.get_model_data(start_date, end_date)
merged_df = processor.interpolate_model_to_obs_locations(model_data, obs_data)
cleaned_df = processor.clean_merged_data(merged_df)
train_df, test_df = processor.split_train_test(cleaned_df)
```

See [training_processor_example.py](../examples/training_processor_example.py) for a complete example.

## Workflow

The unified training processor follows these steps:

1. **Data Retrieval**
   - Retrieve observation data from configured sources
   - Retrieve model data for configured variables/categories

2. **Data Alignment**
   - Interpolate model data to observation locations
   - Align timestamps

3. **Data Cleaning**
   - Remove missing values
   - Apply valid range filters
   - Remove duplicates
   - Sort by site and time

4. **Feature Engineering**
   - Add time-based features (hour, day, month)
   - Create log-transformed variables
   - Calculate bias (observation - model)

5. **Train/Test Split**
   - Either temporal (default) or random
   - Save training and testing datasets

6. **Visualization**
   - Distribution plots
   - Scatter plots of observed vs model
   - Bias distribution
   - Time series plots
   - Correlation heatmaps

## Output Files

The processor generates these output files:

- `{target}_{date_range}_FULL.csv`: Complete processed dataset
- `{target}_{date_range}_TRAIN.csv`: Training dataset
- `{target}_{date_range}_TEST.csv`: Testing dataset
- Visualizations folder with diagnostic plots

## Example Visualizations

The processor creates several visualizations to help understand the data:

1. **Distribution Plots**: Compare observed and model variable distributions
2. **Scatter Plots**: Evaluate model performance vs observations
3. **Bias Plots**: Show the distribution of model errors
4. **Time Series**: Show temporal patterns and train/test splits
5. **Correlation Heatmap**: Show relationships between variables

## Performance Considerations

For large datasets, the processor:
- Processes data in chunks to manage memory
- Supports parallel processing via dask
- Allows customization of chunk sizes and worker counts

## Extending the Processor

You can extend the processor to:
1. Add new feature engineering methods
2. Support new data sources
3. Implement custom data cleaning
4. Create additional visualizations

See the developer documentation for details on extending the processor.

## Troubleshooting

Common issues:
- **Memory errors**: Try reducing chunk_size in config
- **Missing files**: Check paths in config files
- **Empty results**: Check date ranges and spatial bounds

## More Information

- [Unified Atmospheric Data Processing](../step1_input_prepare/UNIFIED_PROCESSING.md)
- [Observation Data Processing](../step2_obs_prepare/OBSERVATION_PROCESSING.md)
