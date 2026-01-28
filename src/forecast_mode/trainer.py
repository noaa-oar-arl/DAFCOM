# ===============
# import library
# ===============

# ================
# main
# ===============
"""
Aero-compliant Model Training Module.
🍃⚡ This module provides the high-level ModelTrainer class for orchestrating ML training.
"""

from typing import Any, List, Optional, Union

import pandas as pd
import xarray as xr

from .models.lstm_model import LSTMPipeline


class ModelTrainer:
    """
    Orchestrator for training machine learning models in DAFCOM.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the trainer with configuration.

        Parameters
        ----------
        config : dict, optional
        """
        self.config = config or {}

    def train_lstm_iterative(
        self,
        data_batches: List[Union[pd.DataFrame, xr.Dataset, xr.DataArray]],
        features: List[str],
        target: str = "log_pm25",
        epochs_per_batch: int = 5,
        model_path: str = "lstm_model.keras",
    ) -> List[Any]:
        """
        Train an LSTM model iteratively on a list of data batches.

        🍃⚡ Aero Protocol: Supports xr.Dataset and xr.DataArray inputs.

        Parameters
        ----------
        data_batches : List[Union[pd.DataFrame, xr.Dataset, xr.DataArray]]
            A list of data batches to train on.
        features : List[str]
            Features to use for training.
        target : str, default 'log_pm25'
            The target variable.
        epochs_per_batch : int, default 5
            Number of epochs for each batch.
        model_path : str, default 'lstm_model.keras'
            Where to save/load the model.

        Returns
        -------
        List[Any]
            A list of history objects from each training iteration.
        """
        pipeline = LSTMPipeline(
            features=features,
            target=target,
            model_path=model_path,
        )

        histories = []
        for i, batch in enumerate(data_batches):
            print(f"--- Training Batch {i + 1}/{len(data_batches)} ---")
            history = pipeline.train_iterative(batch, epochs=epochs_per_batch)
            histories.append(history)

        return histories
