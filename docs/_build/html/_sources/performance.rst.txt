Performance Tuning
===============

This guide provides strategies for optimizing DAFCOM models for both accuracy and computational efficiency.

Model Selection
-------------

**When to Use Direct XGBoost**

- For smaller datasets (thousands to tens of thousands of samples)
- When interpretability is important
- For faster training and inference
- When temporal patterns are less important

**When to Use Two-Stage Pipeline**

- For larger datasets with strong temporal patterns
- For capturing both long-term trends and short-term fluctuations
- When maximum accuracy is required
- When you have sufficient training data and computational resources

Hyperparameter Tuning
-------------------

**XGBoost Hyperparameters**

.. code-block:: python

   import optuna
   from dafcom.utils.tuning import optimize_xgboost

   # Define hyperparameter search space
   param_space = {
       'n_estimators': (50, 500),
       'max_depth': (3, 10),
       'learning_rate': (0.01, 0.3),
       'subsample': (0.5, 1.0),
       'colsample_bytree': (0.5, 1.0),
       'gamma': (0, 5),
       'min_child_weight': (1, 10)
   }

   # Run optimization
   best_params, study = optimize_xgboost(
       train_data=train_data,
       feature_columns=config['feature_columns'],
       target_column=config['target_column'],
       param_space=param_space,
       n_trials=100,
       cv_folds=3,
       metric='rmse',
       direction='minimize',
       timeout=3600  # 1 hour
   )

   print(f"Best parameters: {best_params}")
   print(f"Best score: {study.best_value:.4f}")

**LSTM Hyperparameters**

.. code-block:: python

   from dafcom.utils.tuning import optimize_lstm

   # Define LSTM hyperparameter search space
   lstm_param_space = {
       'units': (16, 256),
       'layers': (1, 3),
       'dropout': (0.0, 0.5),
       'recurrent_dropout': (0.0, 0.5),
       'learning_rate': (0.0001, 0.01),
       'batch_size': (8, 128)
   }

   # Run optimization
   best_lstm_params, lstm_study = optimize_lstm(
       train_data=train_data,
       input_features=pipeline_config['stage1']['input_features'],
       target_feature=pipeline_config['stage1']['target_feature'],
       sequence_length=pipeline_config['stage1']['sequence_length'],
       forecast_horizon=pipeline_config['stage1']['forecast_horizon'],
       param_space=lstm_param_space,
       n_trials=50,
       validation_split=0.2,
       metric='val_loss',
       direction='minimize',
       early_stopping=True,
       patience=10
   )

   print(f"Best LSTM parameters: {best_lstm_params}")

Feature Engineering
----------------

**Feature Importance Analysis**

.. code-block:: python

   from dafcom.utils.feature_selection import analyze_feature_importance

   # Analyze feature importance
   feature_analysis = analyze_feature_importance(
       model=xgb_model,
       data=train_data,
       n_iterations=10,
       cv_folds=5
   )

   # Print top features
   for feature, score in feature_analysis['top_features']:
       print(f"{feature}: {score:.4f}")

   # Select optimal feature subset
   optimal_features = feature_analysis['optimal_features']
   print(f"Optimal feature set: {optimal_features}")

**Feature Transformation**

.. code-block:: python

   from dafcom.utils.preprocessing import transform_features

   # Apply transformations
   transformed_data = transform_features(
       data=train_data,
       transformations={
           'pm25': 'log',
           'temperature': 'standardize',
           'wind_speed': 'min_max',
           'precipitation': 'power',
           'hour': 'cyclic'
       }
   )

Speed Optimization
---------------

**Faster Training**

1. **Reduce Dataset Size**:

   .. code-block:: python

      # Sample data for faster iterations
      sample_size = min(10000, len(train_data))
      sample_data = train_data.sample(sample_size, random_state=42)

      # Quick model training
      quick_metrics = xgb_model.train(sample_data)

2. **Use Early Stopping**:

   .. code-block:: python

      # Configure early stopping
      config['early_stopping_rounds'] = 10
      config['validation_split'] = 0.2

3. **Optimize Number of Threads**:

   .. code-block:: python

      # Set thread count
      config['nthread'] = 4  # Adjust based on CPU cores

4. **GPU Acceleration**:

   .. code-block:: python

      # Enable GPU acceleration for XGBoost
      config['tree_method'] = 'gpu_hist'
      config['gpu_id'] = 0

**Faster Inference**

1. **Batch Processing**:

   .. code-block:: python

      # Process predictions in batches
      batch_size = 1000
      all_predictions = []

      for i in range(0, len(test_data), batch_size):
          batch = test_data.iloc[i:i+batch_size]
          batch_pred = model.predict(batch[model.feature_columns])
          all_predictions.extend(batch_pred)

2. **Model Pruning**:

   .. code-block:: python

      # Prune the model for faster inference
      from dafcom.utils.optimization import prune_model

      pruned_model = prune_model(
          model=xgb_model,
          data=val_data,
          target_accuracy=0.95  # Allow 5% accuracy drop
      )

      # Save pruned model
      pruned_model.save('models/pruned_model.joblib')

Memory Optimization
----------------

For large datasets or limited memory environments:

1. **Use Data Generators**:

   .. code-block:: python

      from dafcom.utils.data import DataGenerator

      # Create data generator
      generator = DataGenerator(
          data_path='large_dataset.csv',
          batch_size=32,
          feature_columns=config['feature_columns'],
          target_column=config['target_column'],
          shuffle=True,
          chunk_size=10000  # Process 10k rows at a time
      )

      # Train using generator
      model.train_with_generator(generator)

2. **Feature Selection**:

   .. code-block:: python

      # Select top N features only
      from dafcom.utils.feature_selection import select_top_features

      top_features = select_top_features(
          data=train_data,
          target=config['target_column'],
          n_features=15,
          method='mutual_info'
      )

      # Update config
      config['feature_columns'] = top_features

3. **Reduce Precision**:

   .. code-block:: python

      # Convert to float32 for memory savings
      train_data = train_data.astype('float32')

Benchmark Results
--------------

Reference benchmark results on standard hardware for different dataset sizes:

.. list-table:: Training Time Benchmarks
   :widths: 15 15 15 15 15 15
   :header-rows: 1

   * - Dataset Size
     - XGBoost (CPU)
     - XGBoost (GPU)
     - LSTM (CPU)
     - LSTM (GPU)
     - Two-Stage (GPU)
   * - 10k samples
     - 5s
     - 3s
     - 30s
     - 15s
     - 18s
   * - 100k samples
     - 30s
     - 10s
     - 5min
     - 1min
     - 1.2min
   * - 1M samples
     - 5min
     - 1min
     - 50min
     - 10min
     - 11min

.. note::

   These benchmarks were measured on a system with an Intel i7-11700K CPU and NVIDIA RTX 3080 GPU.
   Your results may vary based on hardware and specific dataset characteristics.
