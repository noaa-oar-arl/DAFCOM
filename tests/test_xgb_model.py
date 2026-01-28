"""
Tests for XGBoost Pipeline and Iterative Training.
🍃⚡ Aero Protocol: Verifies Eager (NumPy) and Lazy (Dask) backends.
"""

import os

import numpy as np
import pandas as pd

from src.forecast_mode.models.xgb_model import XGBPipeline
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
                "pm25": rng.standard_normal(n_hours),
                "aod": rng.standard_normal(n_hours),
                "pressfc": rng.standard_normal(n_hours),
                "tmp2m": rng.standard_normal(n_hours),
                "spfh2m": rng.standard_normal(n_hours),
                "hpbl": rng.standard_normal(n_hours),
                "ugrd10m": rng.standard_normal(n_hours),
                "vgrd10m": rng.standard_normal(n_hours),
            }
        )
        data.append(df_site)
    return pd.concat(data)


def test_xgb_pipeline_eager_lazy():
    """Verify that XGBPipeline produces identical sequences for Eager and Lazy inputs."""
    features = ["aod", "pressfc"]
    target = "pm25"
    df = create_synthetic_df()

    # Eager (Pandas)
    pipeline_eager = XGBPipeline(
        df=df, features=features, target=target, time_step=20, time_step_short=10, model_path="test_xgb_eager.json"
    )
    X_eager, Y_eager = pipeline_eager.prepare_sequences()

    # Lazy (Xarray/Dask)
    ds = df.set_index(["site_index", "time_utc"]).to_xarray()
    ds = ds.chunk({"time_utc": 50})

    pipeline_lazy = XGBPipeline(
        df=ds, features=features, target=target, time_step=20, time_step_short=10, model_path="test_xgb_lazy.json"
    )
    X_lazy, Y_lazy = pipeline_lazy.prepare_sequences()

    # Assertions
    np.testing.assert_allclose(X_eager, X_lazy)
    np.testing.assert_allclose(Y_eager, Y_lazy)

    # Cleanup
    for f in ["test_xgb_eager.json", "test_xgb_lazy.json"]:
        if os.path.exists(f):
            os.remove(f)


def test_xgb_iterative_training():
    """Verify that iterative training works and loads existing models."""
    features = ["aod", "pressfc"]
    df = create_synthetic_df(n_hours=400)
    df1 = df.iloc[:200]
    df2 = df.iloc[200:]

    model_path = "test_xgb_iterative.json"
    if os.path.exists(model_path):
        os.remove(model_path)

    pipeline = XGBPipeline(features=features, time_step=20, time_step_short=10, model_path=model_path)

    # First batch
    pipeline.train_iterative(df1, num_boost_round=1)
    assert os.path.exists(model_path), "Model should be saved after first iteration"

    # Second batch (should load existing model)
    pipeline.train_iterative(df2, num_boost_round=1)
    assert os.path.exists(model_path), "Model should still exist"

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)


def test_model_trainer_xgb_iterative():
    """Verify ModelTrainer orchestrates iterative XGBoost training."""
    features = ["aod", "pressfc"]
    df = create_synthetic_df(n_hours=400)
    batches = [df.iloc[:200], df.iloc[200:]]

    trainer = ModelTrainer()
    model_path = "trainer_test_xgb.json"
    if os.path.exists(model_path):
        os.remove(model_path)

    models = trainer.train_xgb_iterative(
        data_batches=batches, features=features, num_boost_round_per_batch=1, model_path=model_path
    )

    assert len(models) == 2
    assert os.path.exists(model_path)

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
