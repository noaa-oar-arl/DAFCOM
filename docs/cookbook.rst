Cookbook
========

This cookbook provides practical recipes for common DAFCOM tasks and usage patterns.

Data Preparation
--------------

**Loading and Preprocessing Data**

.. code-block:: python

   import pandas as pd
   import numpy as np
   from dafcom.utils.preprocessing import preprocess_data

   # Load data from CSV
   data = pd.read_csv('raw_data.csv', parse_dates=['datetime'], index_col='datetime')

   # Preprocess the data
   processed_data = preprocess_data(
       data,
       fill_method='interpolate',
       normalize=True,
       log_transform=['PM25']
   )

   # Split into train/validation/test
   train_data = processed_data.loc['2023-01-01':'2023-06-30']
   val_data = processed_data.loc['2023-07-01':'2023-07-31']
   test_data = processed_data.loc['2023-08-01':]

**Creating Time-Based Features**

.. code-block:: python

   from dafcom.utils.preprocessing import add_time_features

   # Add time-based features
   data_with_time = add_time_features(
       data,
       add_hour=True,
       add_day_of_week=True,
       add_day_of_year=True,
       add_month=True,
       add_season=True
   )

Model Training
------------

**Training a Direct XGBoost Model**

.. code-block:: python

   from dafcom.forecast.models.xgboost_model import XGBoostModel
   from dafcom.utils.config import load_config

   # Load configuration
   config = load_config('config/xgboost_model_config.yaml')

   # Initialize model
   xgb_model = XGBoostModel(config)

   # Train model
   metrics = xgb_model.train(train_data)

   # Print training metrics
   print(f"Training RMSE: {metrics['train_rmse']:.4f}")
   print(f"Validation RMSE: {metrics['val_rmse']:.4f}")

   # Save model to file
   model_path = metrics['model_path']
   print(f"Model saved to: {model_path}")

**Training a Two-Stage Pipeline**

.. code-block:: python

   from dafcom.forecast.models.two_stage_pipeline import TwoStageForecastPipeline
   from dafcom.forecast.models.time_series_data import TimeSeriesDataPreparation

   # Load pipeline configuration
   pipeline_config = load_config('config/two_stage_pipeline_config.yaml')

   # Initialize pipeline
   pipeline = TwoStageForecastPipeline(pipeline_config)

   # Prepare time series data
   pipeline.prepare_data(
       data=train_data,
       input_features=pipeline_config['stage1']['input_features'],
       target_feature=pipeline_config['stage1']['target_feature'],
       sequence_length=pipeline_config['stage1']['sequence_length'],
       forecast_horizon=pipeline_config['stage1']['forecast_horizon']
   )

   # Train the pipeline
   metrics = pipeline.train(train_data, val_data)

   # Print metrics for both stages
   print(f"Stage 1 validation RMSE: {metrics['stage1']['val_rmse']:.4f}")
   print(f"Stage 2 validation RMSE: {metrics['stage2']['val_rmse']:.4f}")

Prediction and Evaluation
----------------------

**Making Predictions with XGBoost**

.. code-block:: python

   # Load a trained model
   loaded_model = XGBoostModel(config)
   loaded_model.load(model_path)

   # Make predictions
   predictions = loaded_model.predict(test_data[loaded_model.feature_columns])

   # If the target was log-transformed, convert back
   if loaded_model.target_column.startswith('log_'):
       original_predictions = np.exp(predictions)
       original_actuals = np.exp(test_data[loaded_model.target_column])
   else:
       original_predictions = predictions
       original_actuals = test_data[loaded_model.target_column]

   # Evaluate predictions
   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

   rmse = np.sqrt(mean_squared_error(original_actuals, original_predictions))
   mae = mean_absolute_error(original_actuals, original_predictions)
   r2 = r2_score(original_actuals, original_predictions)

   print(f"Test RMSE: {rmse:.4f}")
   print(f"Test MAE: {mae:.4f}")
   print(f"Test R²: {r2:.4f}")

**Evaluating a Two-Stage Pipeline**

.. code-block:: python

   # Make predictions with the pipeline
   pipeline_predictions = pipeline.predict(test_data)

   # Evaluate pipeline performance
   evaluation_metrics = pipeline.evaluate(test_data)

   print(f"Pipeline RMSE: {evaluation_metrics['rmse']:.4f}")
   print(f"Pipeline MAE: {evaluation_metrics['mae']:.4f}")
   print(f"Pipeline R²: {evaluation_metrics['r2']:.4f}")

Batch Processing
--------------

**Processing Multiple Locations**

.. code-block:: python

   import os
   from dafcom.utils.batch import process_locations

   # Define locations to process
   locations = [
       {'id': 'loc1', 'name': 'Location 1', 'lat': 40.7128, 'lon': -74.0060},
       {'id': 'loc2', 'name': 'Location 2', 'lat': 34.0522, 'lon': -118.2437},
       {'id': 'loc3', 'name': 'Location 3', 'lat': 41.8781, 'lon': -87.6298}
   ]

   # Process all locations
   results = process_locations(
       locations=locations,
       model=xgb_model,
       data_dir='data/',
       output_dir='results/',
       n_jobs=-1  # Use all available cores
   )

   # Summarize results
   for loc_id, metrics in results.items():
       print(f"{loc_id}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

**Temporal Forecasting**

.. code-block:: python

   from dafcom.utils.forecasting import rolling_forecast

   # Perform rolling forecasts
   forecast_results = rolling_forecast(
       model=xgb_model,
       data=data,
       start_date='2023-09-01',
       end_date='2023-09-30',
       forecast_horizon=24,  # Hours
       retrain_frequency=168,  # Hours (weekly retraining)
       training_window=90,  # Days of training data
       feature_columns=config['feature_columns'],
       target_column=config['target_column']
   )

   # Access results
   forecasts = forecast_results['forecasts']
   actuals = forecast_results['actuals']
   metrics = forecast_results['metrics']

Model Deployment
-------------

**Saving Model for Production**

.. code-block:: python

   from dafcom.utils.deployment import package_model

   # Package the model for deployment
   deployment_path = package_model(
       model=xgb_model,
       config=config,
       output_dir='deployment/',
       include_preprocessing=True,
       version='1.0.0',
       metadata={'author': 'Your Name', 'date': '2023-09-01'}
   )

   print(f"Model packaged for deployment at: {deployment_path}")

**Serving Predictions via REST API**

.. code-block:: python

   from dafcom.utils.api import create_prediction_app

   # Create a Flask application for serving predictions
   app = create_prediction_app(
       model_path=deployment_path,
       port=5000,
       debug=False,
       allow_cors=True
   )

   # Run the app
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)

   # Access via: http://localhost:5000/predict with POST request
