Configuration Guide
==================

DAFCOM uses YAML configuration files to control model parameters and behavior. This guide explains the configuration options and provides examples.

XGBoost Model Configuration
--------------------------

The XGBoost model configuration file controls parameters for standalone XGBoost models. Here's an example with explanations:

.. code-block:: yaml

    # Output directory for saving models and results
    output_dir: ./models/xgboost

    # Feature columns used by XGBoost
    feature_columns:
      - v1_blh            # Boundary layer height
      - v2_d2m            # 2m dewpoint temperature
      - v3_e              # Evaporation
      - v4_sp             # Surface pressure
      - v5_t2m            # 2m temperature
      - v6_tp             # Total precipitation
      - v7_u10            # 10m U wind component
      - v8_v10            # 10m V wind component
      - v9_aod            # Aerosol optical depth
      - v10_luc           # Land use category
      - v11_elevation     # Surface elevation
      - v12_population    # Population density
      - v13_ufs_pm25      # UFS PM2.5 predictions
      - hour_utc          # Hour of day (UTC)
      - day_of_year       # Day of year

    # Target column (usually log-transformed for better results)
    target_column: log_pm25

    # XGBoost hyperparameters
    objective: reg:squarederror    # Regression with squared error
    max_depth: 6                   # Maximum tree depth
    learning_rate: 0.1             # Step size shrinkage
    n_estimators: 100              # Number of trees
    subsample: 0.8                 # Subsample ratio of training data
    colsample_bytree: 0.8          # Subsample ratio of columns
    gamma: 0                       # Minimum loss reduction for partition
    min_child_weight: 1            # Minimum sum of instance weight
    early_stopping_rounds: 10      # Stop if no improvement after N rounds

    # Hyperparameter tuning options
    perform_hyperparameter_tuning: false
    hyperparameter_tuning_iterations: 10
    cv_folds: 3
    random_state: 42
    validation_split: 0.2

Two-Stage Pipeline Configuration
------------------------------

The two-stage pipeline configuration controls both the sequential model (LSTM/Transformer) and the XGBoost model. Here's an example:

.. code-block:: yaml

    # Stage 1: Sequential Model Configuration
    stage1:
      # Model type: 'lstm' or 'transformer'
      model_type: lstm

      # Data configuration
      input_features:
        - v1_blh
        - v2_d2m
        # ... other features
      target_feature: log_pm25
      sequence_length: 24          # Hours of historical data to use
      forecast_horizon: 12         # Hours to forecast

      # Model parameters
      model_params:
        units: 64                  # LSTM units
        dropout: 0.2               # Dropout rate
        recurrent_dropout: 0.2     # Recurrent dropout rate
        learning_rate: 0.001       # Learning rate
        epochs: 100                # Maximum epochs
        batch_size: 32             # Batch size
        patience: 10               # Early stopping patience
        validation_split: 0.2      # Validation data ratio

      # Output path for saving the model
      model_path: ./models/lstm

    # Stage 2: XGBoost Configuration
    stage2:
      # All parameters are similar to the standalone XGBoost config
      feature_columns:
        - v1_blh
        - v2_d2m
        # ... other features
        - lstm_predictions         # Add stage 1 predictions as a feature
      target_column: log_pm25
      model_path: ./models/xgboost
      n_estimators: 100
      # ... other XGBoost parameters

Best Practices
------------

1. **Feature Selection**: Include all relevant features that might influence air quality
2. **Log Transformation**: For PM2.5, using log-transformed values often improves model performance
3. **Validation Split**: Always set aside validation data (15-20%) for early stopping
4. **Early Stopping**: Use early stopping to prevent overfitting
5. **Hyperparameter Tuning**: For production models, enable hyperparameter tuning
6. **Sequence Length**: For LSTM models, choose a sequence length that captures relevant temporal patterns
   (typically 24 hours for daily cycles, 168 hours for weekly patterns)
7. **Model Size**: Adjust LSTM units and XGBoost n_estimators based on dataset size
