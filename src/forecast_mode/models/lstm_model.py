"""
Aero-compliant LSTM Pipeline Module.
🍃⚡ This module provides a pipeline for training and evaluating LSTM models for air quality forecasting.
"""

# ===================
# Import Library
# ===================
from datetime import timedelta
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
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
        df: Optional[pd.DataFrame] = None,
        time_step: int = 120,
        time_step_short: int = 96,
        features: Optional[List[str]] = None,
        target: str = "log_pm25",
        scaler_x_path: str = "Scaler_X.save",
        scaler_y_path: str = "Scaler_Y.save",
        model_path: str = "lstm_model.h5",
        test_size: float = 0.2,
        shuffle: bool = False,
    ) -> None:
        """
        Initialize the LSTM pipeline.

        Parameters
        ----------
        csv_path : str, optional
            Path to the CSV data file.
        df : pd.DataFrame, optional
            DataFrame containing training data.
        time_step : int, default 120
            Total time steps for each sequence.
        time_step_short : int, default 96
            Input sequence length for the LSTM.
        features : List[str], optional
            List of feature names.
        target : str, default 'log_pm25'
            Target variable name.
        scaler_x_path : str, default 'Scaler_X.save'
            Path to save/load the features scaler.
        scaler_y_path : str, default 'Scaler_Y.save'
            Path to save/load the target scaler.
        model_path : str, default 'lstm_model.h5'
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

    def load_csv(self) -> pd.DataFrame:
        """
        Load CSV data from path or return the provided DataFrame.

        Returns
        -------
        pd.DataFrame
            The loaded training data.
        """
        if self.df is not None:
            return self.df.copy()
        if not self.csv_path:
            raise ValueError("No csv_path or DataFrame provided")
        return pd.read_csv(self.csv_path)

    def prepare_sequences(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.

        Parameters
        ----------
        df : pd.DataFrame, optional
            The data to prepare. If None, uses loaded data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Features (X) and target (Y) sequences.
        """
        if df is None:
            df = self.load_csv()

        df["time_utc"] = pd.to_datetime(df["time_utc"], format="%Y-%m-%d %H:%M:%S")

        site_list = list(df["site_index"])
        time_list = list(df["time_utc"])

        squence_line_start_list = []
        for i in range(len(site_list)):
            if i + self.time_step <= len(site_list):
                if site_list[i + self.time_step - 1] == site_list[i]:
                    time_start = time_list[i]
                    time_end = time_list[i + self.time_step - 1]
                    if time_start + timedelta(hours=self.time_step) == time_end:
                        squence_line_start_list.append(i)

        if self.features is None:
            raise ValueError("features must be provided (list of column names)")

        data_x = df[self.features[1:]]
        data_y = df[self.features[0]]

        data_y = pd.DataFrame(np.array(data_y).reshape((len(data_y), 1)))
        data_y = data_y.rename(columns={0: self.target})

        self.scaler_x = RobustScaler()
        data_scaled_X = self.scaler_x.fit_transform(data_x)

        self.scaler_y = RobustScaler()
        data_scaled_Y = self.scaler_y.fit_transform(data_y)

        joblib.dump(self.scaler_x, self.scaler_x_path)
        joblib.dump(self.scaler_y, self.scaler_y_path)

        X = []
        Y = []
        for start in squence_line_start_list:
            X.append(data_scaled_X[start : start + self.time_step_short, :])
            Y.append(data_scaled_Y[start + self.time_step_short : start + self.time_step, :])

        self.X = np.array(X)
        self.Y = np.array(Y)

        return self.X, self.Y

    def split_and_reshape(self) -> None:
        """
        Split sequences into training and testing sets and reshape for LSTM.
        """
        if self.X is None or self.Y is None:
            raise RuntimeError("Call prepare_sequences first")

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_size, shuffle=self.shuffle)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test

    def build_model(self, lstm_units: int = 84, dense_units: Tuple[int, int] = (28, 24), dropout: float = 0.2) -> Any:
        """
        Build the LSTM model using Keras.

        Parameters
        ----------
        lstm_units : int, default 84
            Number of units in LSTM layers.
        dense_units : Tuple[int, int], default (28, 24)
            Units for dense layers.
        dropout : float, default 0.2
            Dropout rate.

        Returns
        -------
        keras.models.Sequential
            The compiled model.
        """
        try:
            from keras.layers import LSTM, Dense
            from keras.models import Sequential
        except Exception as exc:
            raise ImportError("Keras/TensorFlow not available.") from exc

        if self.X_train is None:
            raise RuntimeError("Call split_and_reshape first")

        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        model = Sequential(
            [
                LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape, dropout=dropout),
                LSTM(units=lstm_units, return_sequences=False, dropout=dropout),
                Dense(units=dense_units[0]),
                Dense(units=dense_units[1]),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")
        self.model = model
        return model

    def train_model(self, epochs: int = 10, batch_size: int = 48, validation: bool = True, verbose: int = 1) -> Any:
        """
        Train the LSTM model.

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

        Returns
        -------
        keras.callbacks.History
            Training history.
        """
        if self.model is None:
            raise RuntimeError("Call build_model first")
        if self.X_train is None or self.Y_train is None:
            raise RuntimeError("Call split_and_reshape first")

        val = (self.X_test, self.Y_test) if validation and self.X_test is not None else None
        history = self.model.fit(
            self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size, validation_data=val, verbose=verbose
        )
        self.model.save(self.model_path)
        return history

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
        Y_test = self.Y_test.reshape(np.shape(self.Y_test[:, :, 0]))
        Y_predict_reverseScale = self.scaler_y.inverse_transform(Y_predict)
        Y_test_reverseScale = self.scaler_y.inverse_transform(Y_test)

        pd.DataFrame(Y_predict_reverseScale).to_csv(out_predict_csv, index=False)
        pd.DataFrame(Y_test_reverseScale).to_csv(out_test_csv, index=False)

        return Y_predict_reverseScale, Y_test_reverseScale
