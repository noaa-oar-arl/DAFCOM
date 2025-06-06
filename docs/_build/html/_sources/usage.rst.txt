Usage
=====

Basic Usage
----------

DAFCOM provides a simple interface for training and applying forecasting models.

Direct XGBoost Forecasting
-------------------------

The simplest way to use DAFCOM is to train and apply a direct XGBoost model:

.. code-block:: python

   from dafcom.forecast.models.xgboost_model import XGBoostModel
   from dafcom.utils.config import load_config

   # Load configuration
   config_path = "path/to/config.yaml"
   config = load_config(config_path)

   # Initialize model
   model = XGBoostModel(config)

   # Train model
   training_data = pd.read_csv("path/to/training_data.csv")
   metrics = model.train(training_data)

   # Make predictions
   test_data = pd.read_csv("path/to/test_data.csv")
   predictions = model.predict(test_data[model.feature_columns])

   # Evaluate predictions
   from sklearn.metrics import mean_squared_error
   rmse = np.sqrt(mean_squared_error(test_data[model.target_column], predictions))
   print(f"Test RMSE: {rmse:.4f}")

Two-Stage Pipeline
----------------

For more advanced forecasting, use the two-stage pipeline:

.. code-block:: python

   from dafcom.forecast.models.two_stage_pipeline import TwoStageForecastPipeline

   # Load configuration
   pipeline_config_path = "path/to/pipeline_config.yaml"
   pipeline_config = load_config(pipeline_config_path)

   # Initialize pipeline
   pipeline = TwoStageForecastPipeline(pipeline_config)

   # Prepare data
   from dafcom.forecast.models.time_series_data import TimeSeriesDataPreparation
   data = pd.read_csv("path/to/data.csv", parse_dates=["datetime"], index_col="datetime")

   # Split data
   train_data = data.loc["2022":"2023"]
   val_data = data.loc["2024"]
   test_data = data.loc["2025"]

   # Train pipeline
   metrics = pipeline.train(train_data, val_data)

   # Make predictions
   predictions = pipeline.predict(test_data)

   # Evaluate
   evaluation_metrics = pipeline.evaluate(test_data)
   print(f"Test RMSE: {evaluation_metrics['rmse']:.4f}")

Command-line Interface
--------------------

DAFCOM also provides a command-line interface for common tasks:

.. code-block:: bash

   # Train a model
   python -m dafcom.forecast.cli train --config path/to/config.yaml --data path/to/data.csv

   # Make predictions
   python -m dafcom.forecast.cli predict --model path/to/model.joblib --data path/to/data.csv

   # Evaluate a model
   python -m dafcom.forecast.cli evaluate --model path/to/model.joblib --data path/to/data.csv
