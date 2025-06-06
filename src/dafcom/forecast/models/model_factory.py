#!/usr/bin/env python3
"""
DAFCOM Model Factory - Time Series Forecasting Models

This module provides a factory for creating and training different types of models:
1. LSTM-based models for time series forecasting
2. Transformer-based models for time series forecasting

The module supports distributed training using Dask for efficient processing
of large datasets.

Author: GitHub Copilot
Date: June 5, 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Input, GlobalAveragePooling1D, Concatenate, Embedding, TimeDistributed
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
import dask
import dask.array as da
from dask.distributed import Client

# Import utilities from DAFCOM package
from dafcom.utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesModel:
    """Base class for all time series models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base time series model.

        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config
        self.model = None
        self.history = None
        self.feature_columns = config.get('feature_columns', [])
        self.target_columns = config.get('target_columns', [])
        self.sequence_length = config.get('sequence_length', 24)
        self.forecast_horizon = config.get('forecast_horizon', 48)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.validation_split = config.get('validation_split', 0.2)
        self.model_path = config.get('model_path', './models')

        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

    def build_model(self) -> None:
        """
        Build the model architecture.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build_model()")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.

        Args:
            X_train: Training features (shape: [samples, time steps, features])
            y_train: Training targets (shape: [samples, forecast_horizon, targets])
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            self.build_model()

        # Set up callbacks
        model_filename = f"{self.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}.h5"
        model_path = os.path.join(self.model_path, model_filename)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Add TensorBoard callback if log directory is specified
        if 'tensorboard_log_dir' in self.config:
            log_dir = self.config['tensorboard_log_dir']
            os.makedirs(log_dir, exist_ok=True)
            callbacks.append(TensorBoard(log_dir=log_dir))

        # Train the model
        logger.info(f"Training {self.__class__.__name__} model...")

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.config.get('epochs', 100),
                batch_size=self.batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.config.get('epochs', 100),
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1
            )

        logger.info(f"Model training completed. Best model saved to {model_path}")
        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            X: Input features (shape: [samples, time steps, features])

        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")

        return self.model.predict(X)

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.

        Args:
            path: Optional path to save the model

        Returns:
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Use provided path or default from config
        save_path = path or os.path.join(
            self.model_path,
            f"{self.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}.h5"
        )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the model
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

        # Save feature and target information
        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'config': self.config,
            'model_type': self.__class__.__name__
        }

        metadata_path = save_path.replace('.h5', '_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        logger.info(f"Model metadata saved to {metadata_path}")

        return save_path

    @classmethod
    def load(cls, path: str) -> 'TimeSeriesModel':
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance
        """
        # Load metadata
        metadata_path = path.replace('.h5', '_metadata.joblib')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

        metadata = joblib.load(metadata_path)

        # Create instance with saved config
        instance = cls(metadata['config'])

        # Load model weights
        instance.model = tf.keras.models.load_model(path)

        # Restore metadata attributes
        instance.feature_columns = metadata['feature_columns']
        instance.target_columns = metadata['target_columns']
        instance.sequence_length = metadata['sequence_length']
        instance.forecast_horizon = metadata['forecast_horizon']

        logger.info(f"Model loaded from {path}")
        return instance


class LSTMModel(TimeSeriesModel):
    """LSTM-based time series forecasting model."""

    def build_model(self) -> None:
        """
        Build the LSTM model architecture.
        """
        n_features = len(self.feature_columns)
        n_outputs = len(self.target_columns)

        # Get LSTM-specific parameters from config
        lstm_units = self.config.get('lstm_units', [64, 32])
        dropout_rate = self.config.get('dropout_rate', 0.2)

        # Build the sequential model
        model = Sequential()

        # Add LSTM layers with dropout
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=(self.sequence_length, n_features)
        ))
        model.add(Dropout(dropout_rate))

        # Add additional LSTM layers if specified
        for i in range(1, len(lstm_units)):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units=lstm_units[i], return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))

        # Output layers
        model.add(Dense(
            units=self.forecast_horizon * n_outputs,
            activation='linear'
        ))

        # Reshape output to match forecast horizon and number of target features
        model.add(tf.keras.layers.Reshape((self.forecast_horizon, n_outputs)))

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self.config.get('loss', 'mse'),
            metrics=[self.config.get('metrics', ['mae'])]
        )

        # Print model summary
        model.summary()

        self.model = model


class TransformerModel(TimeSeriesModel):
    """Transformer-based time series forecasting model."""

    def build_model(self) -> None:
        """
        Build the Transformer model architecture.
        """
        n_features = len(self.feature_columns)
        n_outputs = len(self.target_columns)

        # Get transformer-specific parameters
        head_size = self.config.get('head_size', 256)
        num_heads = self.config.get('num_heads', 4)
        ff_dim = self.config.get('ff_dim', 4)
        num_transformer_blocks = self.config.get('num_transformer_blocks', 2)
        mlp_units = self.config.get('mlp_units', [128, 64])
        dropout_rate = self.config.get('dropout_rate', 0.2)
        mlp_dropout = self.config.get('mlp_dropout', 0.2)

        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Multi-head attention
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            x = Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Dense(ff_dim, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            return x + res

        # Build the model
        inputs = Input(shape=(self.sequence_length, n_features))
        x = inputs

        # Add transformer blocks
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(
                x, head_size, num_heads, ff_dim, dropout_rate
            )

        # Add MLP layers for output
        x = LayerNormalization(epsilon=1e-6)(x)
        x = GlobalAveragePooling1D()(x)

        for dim in mlp_units:
            x = Dense(dim, activation='relu')(x)
            x = Dropout(mlp_dropout)(x)

        # Output layer
        outputs = Dense(self.forecast_horizon * n_outputs)(x)
        outputs = tf.keras.layers.Reshape((self.forecast_horizon, n_outputs))(outputs)

        # Create and compile the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self.config.get('loss', 'mse'),
            metrics=self.config.get('metrics', ['mae'])
        )

        # Print model summary
        model.summary()

        self.model = model


def create_model(model_type: str, config: Dict[str, Any]) -> TimeSeriesModel:
    """
    Factory function to create a time series model.

    Args:
        model_type: Type of model to create ('lstm' or 'transformer')
        config: Dictionary containing model configuration

    Returns:
        Instantiated model of the requested type

    Raises:
        ValueError: If an invalid model type is specified
    """
    model_type = model_type.lower()

    if model_type == 'lstm':
        return LSTMModel(config)
    elif model_type == 'transformer':
        return TransformerModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         f"Supported types are 'lstm' and 'transformer'.")
