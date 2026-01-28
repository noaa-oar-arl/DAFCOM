"""
Aero-compliant LSTM Pipeline Module.
🍃⚡ This module provides a pipeline for training and evaluating LSTM models for air quality forecasting.
"""

# ===================
# Import Library
# ===================
import os
from datetime import timedelta
from typing import Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# ===========================
# Define Class and Functions
# ===========================
class LSTMPipeline:
    """
    LSTM Training Pipeline for air quality forecasting.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        df: Optional[Union[pd.DataFrame, xr.Dataset, xr.DataArray]] = None,
        time_step: int = 168,
        time_step_short: int = 96,
        features: Optional[List[str]] = None,
        target: str = "log_pm25",
        scaler_x_path: str = "Scaler_X.save",
        scaler_y_path: str = "Scaler_Y.save",
        model_path: str = "lstm_model.keras",
        test_size: float = 0.2,
        shuffle: bool = False,
    ) -> None:
        """
        Initialize the LSTM pipeline.

        Parameters
        ----------
        csv_path : str, optional
            Path to the CSV data file.
        df : pd.DataFrame or xr.Dataset or xr.DataArray, optional
            Data containing training samples.
        time_step : int, default 168
            Total time steps (Input + Forecast). 168 = 96h input + 72h forecast.
        time_step_short : int, default 96
            Input sequence length for the LSTM (4 days = 96h).
        features : List[str], optional
            List of feature names.
        target : str, default 'log_pm25'
            Target variable name.
        scaler_x_path : str, default 'Scaler_X.save'
            Path to save/load the features scaler.
        scaler_y_path : str, default 'Scaler_Y.save'
            Path to save/load the target scaler.
        model_path : str, default 'lstm_model.keras'
            Path to save the trained model.
        test_size : float, default 0.2
            Proportion of data to use for testing.
        shuffle : bool, default False
            Whether to shuffle the data before splitting.
        """
        self.csv_path = csv_path
        self.df = df
        self.time_step = time_step
        self.time_step_short = time_step_short
        self.features = features
        self.target = target
        self.scaler_x_path = scaler_x_path
        self.scaler_y_path = scaler_y_path
        self.model_path = model_path
        self.test_size = test_size
        self.shuffle = shuffle

        self.scaler_x: Optional[RobustScaler] = None
        self.scaler_y: Optional[RobustScaler] = None
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None
        self.Y_test: Optional[np.ndarray] = None
        self.model: Any = None

    def prepare_sequences(
        self, df: Optional[Union[pd.DataFrame, xr.Dataset, xr.DataArray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.

        Parameters
        ----------
        df : pd.DataFrame or xr.Dataset or xr.DataArray, optional
            The data to prepare. If None, uses loaded data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Features (X) and target (Y) sequences.
        """
        if df is None:
            df = self.df if self.df is not None else self.csv_path

        if isinstance(df, str):
            df = pd.read_csv(df)
        elif isinstance(df, (xr.Dataset, xr.DataArray)):
            # 🍃⚡ Aero Protocol: Convert to dataframe only when needed for non-Xarray backend.
            df = df.to_dataframe().reset_index()
        elif isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            raise ValueError("No data provided or unsupported format")

        df["time_utc"] = pd.to_datetime(df["time_utc"], format="%Y-%m-%d %H:%M:%S")

        site_list = list(df["site_index"])
        time_list = list(df["time_utc"])

        squence_line_start_list = []
        for i in range(len(site_list)):
            if i + self.time_step <= len(site_list):
                if site_list[i + self.time_step - 1] == site_list[i]:
                    time_start = time_list[i]
                    time_end = time_list[i + self.time_step - 1]
                    if time_start + timedelta(hours=self.time_step - 1) == time_end:
                        squence_line_start_list.append(i)

        if self.features is None:
            raise ValueError("features must be provided (list of column names)")

        data_x = df[self.features[1:]]
        data_y = df[self.features[0]]

        data_y = pd.DataFrame(np.array(data_y).reshape((len(data_y), 1)))
        data_y = data_y.rename(columns={0: self.target})

        # Load or fit scalers to ensure consistency across iterative training batches.
        if self.scaler_x is None:
            if os.path.exists(self.scaler_x_path):
                self.scaler_x = joblib.load(self.scaler_x_path)
            else:
                self.scaler_x = RobustScaler()
                self.scaler_x.fit(data_x)
                joblib.dump(self.scaler_x, self.scaler_x_path)

        if self.scaler_y is None:
            if os.path.exists(self.scaler_y_path):
                self.scaler_y = joblib.load(self.scaler_y_path)
            else:
                self.scaler_y = RobustScaler()
                self.scaler_y.fit(data_y)
                joblib.dump(self.scaler_y, self.scaler_y_path)

        data_scaled_X = self.scaler_x.transform(data_x)
        data_scaled_Y = self.scaler_y.transform(data_y)

        X = []
        Y = []
        for start in squence_line_start_list:
            X.append(data_scaled_X[start : start + self.time_step_short, :])
            Y.append(data_scaled_Y[start + self.time_step_short : start + self.time_step, :])

        self.X = np.array(X)
        self.Y = np.array(Y)

        return self.X, self.Y

    def split_and_reshape(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None) -> None:
        """
        Split sequences into training and testing sets and reshape for LSTM.

        Parameters
        ----------
        X : np.ndarray, optional
            Features sequences. If None, uses self.X.
        Y : np.ndarray, optional
            Target sequences. If None, uses self.Y.
        """
        if X is not None:
            self.X = X
        if Y is not None:
            self.Y = Y

        if self.X is None or self.Y is None:
            raise RuntimeError("Call prepare_sequences first or provide X and Y")

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_size, shuffle=self.shuffle)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        # Reshape Y to (batch, forecast_length)
        # 🍃⚡ Aero Protocol: Maintain consistent shapes for the ML backend.
        Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1]))
        Y_test = Y_test.reshape((Y_test.shape[0], Y_test.shape[1]))

        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test

    def load_existing_model(self) -> Any:
        """
        Load an existing model from self.model_path if it exists.

        Returns
        -------
        keras.models.Sequential or None
            The loaded model, or None if not found.
        """
        try:
            import keras
        except ImportError as exc:
            raise ImportError("Keras/TensorFlow not available.") from exc

        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            return self.model
        return None

    def build_model(
        self,
        lstm_units: int = 84,
        dense_units: Optional[int] = None,
        dropout: float = 0.2,
        force_rebuild: bool = False,
    ) -> Any:
        """
        Build the LSTM model using Keras.

        Parameters
        ----------
        lstm_units : int, default 84
            Number of units in LSTM layers.
        dense_units : int, optional
            Units for the intermediate dense layer.
        dropout : float, default 0.2
            Dropout rate.
        force_rebuild : bool, default False
            Whether to rebuild the model even if it exists.

        Returns
        -------
        keras.models.Sequential
            The compiled model.
        """
        if not force_rebuild and self.load_existing_model() is not None:
            return self.model

        try:
            from keras.layers import LSTM, Dense, Input
            from keras.models import Sequential
        except ImportError as exc:
            raise ImportError("Keras/TensorFlow not available.") from exc

        if self.X_train is None:
            raise RuntimeError("Call split_and_reshape first")

        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        forecast_length = self.time_step - self.time_step_short
        intermediate_dense = dense_units if dense_units is not None else (forecast_length + lstm_units) // 2

        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(units=lstm_units, return_sequences=True, dropout=dropout),
                LSTM(units=lstm_units, return_sequences=False, dropout=dropout),
                Dense(units=intermediate_dense, activation="relu"),
                Dense(units=forecast_length),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")
        self.model = model
        return model

    def train_model(
        self,
        epochs: int = 10,
        batch_size: int = 48,
        validation: bool = True,
        verbose: int = 1,
        checkpoint: bool = True,
    ) -> Any:
        """
        Train the LSTM model. Supports continual learning by using existing model if available.

        Parameters
        ----------
        epochs : int, default 10
            Number of training epochs.
        batch_size : int, default 48
            Batch size.
        validation : bool, default True
            Whether to use validation data.
        verbose : int, default 1
            Verbosity level.
        checkpoint : bool, default True
            Whether to save model checkpoints.

        Returns
        -------
        keras.callbacks.History
            Training history.
        """
        if self.model is None:
            self.build_model()

        if self.X_train is None or self.Y_train is None:
            raise RuntimeError("Call split_and_reshape first")

        # Determine if we have valid validation data
        has_val = validation and self.X_test is not None and len(self.X_test) > 0
        val = (self.X_test, self.Y_test) if has_val else None

        callbacks = []
        if checkpoint:
            try:
                from keras.callbacks import ModelCheckpoint

                # Only save best if validation data is present
                callbacks.append(ModelCheckpoint(filepath=self.model_path, save_best_only=has_val, verbose=verbose))
            except ImportError:
                pass
        history = self.model.fit(
            self.X_train,
            self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val,
            verbose=verbose,
            callbacks=callbacks,
        )

        # Ensure model is saved even if no checkpointing occurred
        self.model.save(self.model_path)
        return history

    def train_iterative(self, df: Union[pd.DataFrame, xr.Dataset, xr.DataArray], epochs: int = 5) -> Any:
        """
        Perform one iteration of training on a new batch of data.

        Parameters
        ----------
        df : pd.DataFrame or xr.Dataset or xr.DataArray
            The new batch of data.
        epochs : int, default 5
            Number of epochs for this iteration.

        Returns
        -------
        keras.callbacks.History
            Training history.
        """
        X, Y = self.prepare_sequences(df)
        self.split_and_reshape(X, Y)
        return self.train_model(epochs=epochs)

    def plot_history(self, history: Any, title: str = "Model Training History", out_path: str = "training_history.png") -> str:
        """
        Plot training and validation loss from history.

        🍃⚡ Aero Protocol Rule 3: Track A (Publication).

        Parameters
        ----------
        history : keras.callbacks.History
            The training history object.
        title : str, default 'Model Training History'
            The plot title.
        out_path : str, default 'training_history.png'
            Where to save the plot.

        Returns
        -------
        str
            The path to the saved figure.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Train Loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return out_path

    def predict_and_save(
        self, out_predict_csv: str = "Y_predict_original.csv", out_test_csv: str = "Y_test_original.csv"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and save results to CSV.

        Parameters
        ----------
        out_predict_csv : str, default 'Y_predict_original.csv'
            Output path for predictions.
        out_test_csv : str, default 'Y_test_original.csv'
            Output path for actual values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Inverse-scaled predictions and actual values.
        """
        if self.model is None:
            raise RuntimeError("Call build_model and train_model before predict")
        if self.X_test is None:
            raise RuntimeError("No X_test available")

        Y_predict = self.model.predict(self.X_test)
        Y_test = self.Y_test  # Already reshaped in split_and_reshape
        Y_predict_reverseScale = self.scaler_y.inverse_transform(Y_predict)
        Y_test_reverseScale = self.scaler_y.inverse_transform(Y_test)

        pd.DataFrame(Y_predict_reverseScale).to_csv(out_predict_csv, index=False)
        pd.DataFrame(Y_test_reverseScale).to_csv(out_test_csv, index=False)

        return Y_predict_reverseScale, Y_test_reverseScale
