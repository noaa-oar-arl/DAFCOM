"""
Tests for LSTM Pipeline and Iterative Training.
🍃⚡ Aero Protocol: Verifies Eager (NumPy) and Lazy (Dask) backends.
"""

import os

import numpy as np
import pandas as pd

from src.forecast_mode.models.lstm_model import LSTMPipeline
from src.forecast_mode.trainer import ModelTrainer


def create_synthetic_df(n_sites=1, n_hours=200):
    """Create synthetic training data."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    data = []
    for site in range(n_sites):
        df_site = pd.DataFrame(
            {
                "site_index": site,
                "time_utc": times,
                "log_pm25": rng.standard_normal(n_hours),
                "feature1": rng.standard_normal(n_hours),
                "feature2": rng.standard_normal(n_hours),
            }
        )
        data.append(df_site)
    return pd.concat(data)


def test_lstm_pipeline_eager_lazy():
    """Verify that LSTMPipeline produces identical sequences for Eager and Lazy inputs."""
    features = ["log_pm25", "feature1", "feature2"]
    target = "log_pm25"
    df = create_synthetic_df()

    # Eager (Pandas)
    pipeline_eager = LSTMPipeline(
        df=df, features=features, target=target, time_step=20, time_step_short=10, model_path="test_model_eager.keras"
    )
    X_eager, Y_eager = pipeline_eager.prepare_sequences()

    # Lazy (Xarray/Dask)
    ds = df.set_index(["site_index", "time_utc"]).to_xarray()
    ds = ds.chunk({"time_utc": 50})

    pipeline_lazy = LSTMPipeline(
        df=ds, features=features, target=target, time_step=20, time_step_short=10, model_path="test_model_lazy.keras"
    )
    X_lazy, Y_lazy = pipeline_lazy.prepare_sequences()

    # Assertions
    np.testing.assert_allclose(X_eager, X_lazy)
    np.testing.assert_allclose(Y_eager, Y_lazy)

    # Cleanup
    for f in ["test_model_eager.keras", "test_model_lazy.keras", "Scaler_X.save", "Scaler_Y.save"]:
        if os.path.exists(f):
            os.remove(f)


def test_lstm_iterative_training():
    """Verify that iterative training works and loads existing models."""
    features = ["log_pm25", "feature1", "feature2"]
    df = create_synthetic_df(n_hours=400)
    df1 = df.iloc[:200]
    df2 = df.iloc[200:]

    model_path = "test_iterative.keras"
    if os.path.exists(model_path):
        os.remove(model_path)

    pipeline = LSTMPipeline(features=features, time_step=20, time_step_short=10, model_path=model_path)

    # First batch
    pipeline.train_iterative(df1, epochs=1)
    assert os.path.exists(model_path), "Model should be saved after first iteration"

    # Second batch (should load existing model)
    pipeline.train_iterative(df2, epochs=1)
    assert os.path.exists(model_path), "Model should still exist"

    # Cleanup
    for f in [model_path, "Scaler_X.save", "Scaler_Y.save"]:
        if os.path.exists(f):
            os.remove(f)


def test_model_trainer_lstm_iterative():
    """Verify ModelTrainer orchestrates iterative LSTM training."""
    features = ["log_pm25", "feature1", "feature2"]
    df = create_synthetic_df(n_hours=400)
    batches = [df.iloc[:200], df.iloc[200:]]

    trainer = ModelTrainer()
    model_path = "trainer_test.keras"
    if os.path.exists(model_path):
        os.remove(model_path)

    histories = trainer.train_lstm_iterative(data_batches=batches, features=features, epochs_per_batch=1, model_path=model_path)

    assert len(histories) == 2
    assert os.path.exists(model_path)

    # Cleanup
    for f in [model_path, "Scaler_X.save", "Scaler_Y.save"]:
        if os.path.exists(f):
            os.remove(f)
