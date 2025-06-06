# NOAA's DynAmic Forecasting of Air Composition using Optimized Machine Learning

DAFCOM is a comprehensive Python package for air quality forecasting, combining state-of-the-art machine learning models with extensive data processing capabilities.

## Features

- **Two-Stage Forecasting Pipeline**: Sequential models (LSTM/Transformer) combined with XGBoost for bias correction
- **Direct XGBoost Forecasting**: Standalone XGBoost models for rapid forecasting
- **Comprehensive Data Processing**: Tools for preparing and transforming air quality data
- **Visualization Tools**: Built-in capabilities for forecast evaluation
- **Performance Metrics**: Implementation of key metrics for evaluating forecast quality

## Installation

```bash
# Install from the repository
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Direct XGBoost Example

```python
from dafcom.forecast.models.xgboost_model import XGBoostModel
from dafcom.utils.config import load_config

# Load configuration and initialize model
config = load_config("path/to/config.yaml")
model = XGBoostModel(config)

# Train and evaluate
model.train(training_data)
predictions = model.predict(test_data[model.feature_columns])
```

### Command-Line Interface

```bash
# Run direct XGBoost example
python -m dafcom.forecast.examples.direct_xgboost_example --config path/to/config.yaml
```

## Documentation

Comprehensive API documentation is available in the `docs` directory. To build the documentation:

```bash
cd docs
bash build_docs.sh
```

Then open `docs/_build/html/index.html` in your web browser.

## Testing

Run the test suite with:

```bash
# Run with unittest
python run_tests.py

# Run with pytest and generate coverage reports
python run_tests.py --pytest
```

## Major Goals

* Goal 1: build 1st version workflow
  - [x] step 1: process all inputs to be daily, 1 by 1km for June, July, Aug, 2023
  - [x] step 2: build initial forecast & downscaling Machine learning model with 4 cases: 1) train 2023-06, predict 2023-07; 2) train 2023-07, predict 2023-08; 3) train 2023-06, predict 2023-08; 4) train 2023-06&07, predict 2023-08
* Goal 2: build 2nd version workflow
  - [x] step 1: process all inputs for training with nearest 30 days, and predict only 1 day. repeat for predicting 2023-07-01 to 2023-08-31.
  - [x] step 2: analyze ML model performance, and compared to Goal 1.
  - [x] step 3: now we use 30 days to train, update to determine the optimal time-scale/length of training.
* Goal 3: Package organization and code quality
  - [x] step 1: reorganize into proper Python package structure
  - [x] step 2: implement comprehensive error handling and logging
  - [x] step 3: create unified example scripts for various forecasting approaches
  - [x] step 4: add comprehensive unit tests
  - [x] step 5: create API documentation


### Time line of Funding

Timeline or finish date at the miminum
