"""Reproducible site evaluations and diagnostic plots."""

import json
import logging

import numpy as np
import pandas as pd

from .storage import atomic_path, product


def metrics(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    if observed.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("Evaluation arrays must be paired one-dimensional series")
    if len(observed) == 0 or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Evaluation requires nonempty finite paired values")
    correlation = None
    if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    else:
        logging.warning("Correlation undefined for singleton/constant evaluation series")
    error = predicted - observed
    return {
        "count": len(observed),
        "correlation": correlation,
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_bias": float(np.mean(error)),
        "std_model": float(np.std(predicted)),
        "std_obs": float(np.std(observed)),
    }


def evaluate(config, domain):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.concat(
        [
            pd.read_csv(
                product(config, "site_xgb", domain=domain, day=day, suffix="csv"),
                dtype={"site_index": str},
            )
            for day in config.models.prediction.days()
        ],
        ignore_index=True,
    )
    columns = ["xgb_predictions_pm25", "lstm_pm25", "ufs_pm25"]
    if config.baseline_variable:
        columns.append("bias_correction_pm25")
    result = {name: metrics(frame.obs_pm25, frame[name]) for name in columns}
    with atomic_path(product(config, "evaluation", domain=domain, suffix="json")) as temporary:
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    figure, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
    for axis, name in zip(axes[0], columns, strict=True):
        axis.scatter(frame.obs_pm25, frame[name], s=2, alpha=0.3)
        axis.set(xlabel="Observed PM2.5", ylabel=name, title=f"RMSE={result[name]['rmse']:.3f}")
    figure.tight_layout()
    try:
        with atomic_path(product(config, "evaluation", domain=domain, suffix="png")) as temporary:
            figure.savefig(temporary)
    finally:
        plt.close(figure)
