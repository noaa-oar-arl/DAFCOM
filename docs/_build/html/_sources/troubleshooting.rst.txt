Troubleshooting
==============

This guide covers common issues and their solutions when working with DAFCOM.

Installation Issues
-----------------

**Missing Dependencies**

If you encounter errors about missing dependencies:

.. code-block:: bash

   pip install -e .[dev]

**Tensorflow Installation Problems**

For GPU support or compatibility issues with TensorFlow:

.. code-block:: bash

   # CPU-only installation
   pip install tensorflow==2.8.0

   # GPU support
   pip install tensorflow-gpu==2.8.0

Data Processing Issues
--------------------

**Missing Data Columns**

If you receive a ``ValueError`` about missing feature columns:

1. Check that all columns listed in your configuration file exist in your dataset
2. Verify column names match exactly (case sensitive)
3. Run this to display available columns:

   .. code-block:: python

      import pandas as pd
      data = pd.read_csv('your_data.csv')
      print(data.columns.tolist())

**NaN or Infinite Values**

For errors related to NaN or infinite values:

1. Check your data preprocessing steps
2. Clean your data with:

   .. code-block:: python

      # Replace NaN with median values
      data = data.fillna(data.median())

      # Drop rows with any NaN
      data = data.dropna()

Training Issues
-------------

**Out of Memory Errors**

If your model training crashes due to memory issues:

1. Reduce batch size in your configuration
2. Use fewer features or samples
3. For LSTM models, reduce the sequence length or number of units

**Poor Model Performance**

If your model produces poor forecasts:

1. Log-transform highly skewed variables like PM2.5
2. Check for data leakage in your validation set
3. Try different feature combinations
4. Increase the model complexity (more units for LSTM, more estimators for XGBoost)
5. Ensure your training data is representative of test conditions

**Training Takes Too Long**

To speed up model training:

1. Reduce the number of features
2. Use a smaller subset of the data for initial experiments
3. Decrease the maximum number of epochs or estimators
4. Use early stopping with a smaller patience value
5. Use a faster device (GPU if available)

Prediction Issues
---------------

**Model Loading Errors**

If you encounter errors when loading a saved model:

1. Ensure you're using the same version of libraries as during training
2. Verify the model file exists and is not corrupted
3. Check you're using the correct model path

**Incorrect Prediction Shapes**

For shape mismatch errors during prediction:

1. Verify that your input data has the same features as the training data
2. For LSTM models, ensure your input sequence has the correct shape

Logging and Debugging
------------------

To increase logging verbosity for better debugging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

Common Error Messages
------------------

**"Target column 'x' not found in data"**

Ensure your data contains the target column specified in your configuration.

**"ValueError: Input contains NaN, infinity or a value too large"**

Clean your data by removing or imputing NaN values before training.

**"Memory Error" or "ResourceExhaustedError"**

Reduce batch size or model complexity, or use a machine with more memory.

Getting Help
-----------

If you continue experiencing issues:

1. Check the GitHub Issues page for similar problems
2. Provide a minimal reproducible example when seeking help
3. Include error messages and the versions of your dependencies
