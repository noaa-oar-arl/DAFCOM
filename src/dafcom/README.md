# DAFCOM: Data Assimilation and Forecasting for Composition Modeling

DAFCOM is a comprehensive Python package for air quality forecasting that includes data preprocessing, machine learning model training, forecasting, bias correction, and validation tools.

## Package Structure

```
src/dafcom/
├── __init__.py           # Main package initialization
├── forecast/             # Forecasting module
│   ├── __init__.py       # Forecasting module initialization
│   ├── cli.py            # Command-line interface
│   ├── models/           # Machine learning models
│   │   ├── __init__.py   # Models module initialization
│   │   ├── model_factory.py    # Model factory for LSTM and Transformer models
│   │   ├── time_series_data.py # Time series data preparation
│   │   ├── training.py   # Model training utilities
│   │   ├── xgboost_model.py    # XGBoost models for bias correction
│   │   └── two_stage_pipeline.py # Pipeline combining sequence models with XGBoost
│   ├── processor/        # Data processing components
│   │   ├── __init__.py   # Processor module initialization
│   │   └── unified_processor.py # Unified data processor
│   └── examples/         # Example scripts
│       └── train_forecasting_models.py # Example for training models
└── utils/                # Utility functions
    ├── __init__.py       # Utilities module initialization
    ├── config.py         # Configuration utilities
    └── validation.py     # Data validation utilities
```

## Installation

To install DAFCOM from the source code:

```bash
git clone https://github.com/DAFCOM/DAFCOM.git
cd DAFCOM
pip install -e .
```

## Quick Start

### Data Processing

To process data for model training:

```python
from dafcom.forecast.processor import UnifiedTrainingDataProcessor

# Initialize the processor with a configuration file
processor = UnifiedTrainingDataProcessor("path/to/config.yaml")

# Process the data
train_data, val_data, test_data = processor.run_full_processing()
```

### Model Training

DAFCOM supports different model types for forecasting:

#### LSTM or Transformer Models

```python
from dafcom.forecast.models.training import train_model
from dafcom.utils.config import load_config

# Load configuration
config = load_config("path/to/model_config.yaml")

# Train model
model, metrics = train_model(config, data, model_type="lstm")  # or "transformer"
```

#### XGBoost Model for Bias Correction

```python
from dafcom.forecast.models.xgboost_model import XGBoostModel

# Create and train model
model = XGBoostModel(config)
metrics = model.train(data)
```

#### Two-Stage Pipeline (LSTM/Transformer + XGBoost)

```python
from dafcom.forecast.models.two_stage_pipeline import TwoStageForecastPipeline

# Create and train pipeline
pipeline = TwoStageForecastPipeline(config)
metrics = pipeline.train(data)

# Make predictions with the complete pipeline
predictions = pipeline.predict(new_data)
```

### Command-Line Interface

DAFCOM provides a command-line interface for common operations:

```bash
# Process data
python -m dafcom.forecast.cli process --config path/to/config.yaml

# Train models - supports multiple model types
python -m dafcom.forecast.cli train --config path/to/config.yaml --data path/to/data.csv --model-type lstm
python -m dafcom.forecast.cli train --config path/to/config.yaml --data path/to/data.csv --model-type transformer
python -m dafcom.forecast.cli train --config path/to/config.yaml --data path/to/data.csv --model-type xgboost
python -m dafcom.forecast.cli train --config path/to/config.yaml --data path/to/data.csv --model-type two-stage

# Evaluate model
python -m dafcom.forecast.cli evaluate --model path/to/model.h5 --data path/to/test_data.csv --scalers path/to/scalers --config path/to/eval_config.yaml
```

## Examples

The `examples` directory contains ready-to-use examples showing how to work with DAFCOM:

```bash
# Run the model training example
python -m dafcom.forecast.examples.train_forecasting_models
```

## Configuration Files

DAFCOM uses YAML configuration files for various components:

1. **Data Processing Configuration**: Configure data sources, cleaning operations, etc.
2. **Model Configuration**: Configure model hyperparameters, training settings, etc.
3. **Evaluation Configuration**: Configure metrics, visualization settings, etc.

See the `examples` directory for sample configuration files.

## License

MIT
