"""Shared station/grid LSTM and XGBoost forecasting contracts."""

from __future__ import annotations

import json
import logging
import math
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

from .features import grid_features, history_start, table
from .storage import atomic_path, product, write_csv, write_netcdf


def artifact(config, domain, name, suffix):
    return product(config, "models", domain=domain, variable=name, suffix=suffix)


def source_artifact(config, domain, name, suffix):
    if config.model_source is not None:
        return config.model_source / domain / f"{name}.{suffix}"
    return artifact(config, domain, name, suffix)


def science_contract(config, domain):
    return {
        "domain": config.domains[domain].model_dump(mode="json", exclude={"tile_shape"}),
        "variables": {
            name: spec.model_dump(mode="json", exclude={"source", "cycle_day_offsets", "leads"})
            for name, spec in config.variables.items()
        },
        "cycle_hour": config.cycle_hour,
        "distance_reference": config.distance_reference,
    }


def keras_backend():
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("LSTM stages require TensorFlow; install dafcom[ml]") from exc
    return keras


def window_starts(frame, hours):
    """Every internal hour must be contiguous; endpoint checks are insufficient."""
    if hours < 1:
        raise ValueError("Window hours must be positive")
    if len(frame) < hours:
        return np.array([], dtype=int)
    time = frame.time_utc.to_numpy()
    same_site = frame.site_index.to_numpy()[1:] == frame.site_index.to_numpy()[:-1]
    contiguous = (np.diff(time) == np.timedelta64(1, "h")) & same_site
    breaks = np.concatenate(([0], np.cumsum(~contiguous)))
    starts = np.arange(len(frame) - hours + 1)
    return starts[(breaks[starts + hours - 1] - breaks[starts]) == 0]


def transformed_target(values, spec):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (spec.target_transform == "log" and (values <= 0).any()):
        raise ValueError("LSTM target must be finite and strictly positive for log transform")
    return np.log(values) if spec.target_transform == "log" else values


def concentration(values, spec):
    result = np.exp(values) if spec.target_transform == "log" else values
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite LSTM concentration after inverse transform")
    return result


def train_lstm(config, domain):
    keras = keras_backend()
    import tensorflow as tf

    spec = config.models.lstm
    tf.keras.utils.set_random_seed(spec.seed)
    period = config.models.lstm_training
    frame = table(config, domain, period.start, period.end)
    length = spec.lookback_hours + spec.forecast_hours
    starts = window_starts(frame, length)
    cutoff = pd.Timestamp(period.start) + (
        pd.Timestamp(period.end) - pd.Timestamp(period.start)
    ) * (1 - spec.validation_fraction)
    training = starts[frame.time_utc.iloc[starts + length - 1].to_numpy() < cutoff.to_datetime64()]
    validation = starts[frame.time_utc.iloc[starts].to_numpy() >= cutoff.to_datetime64()]
    if not len(training) or not len(validation):
        raise ValueError("LSTM period too short for purged chronological train/validation windows")
    data_x = frame[spec.features].to_numpy(dtype=np.float32)
    data_y = transformed_target(frame.obs_pm25, spec).reshape(-1, 1)
    if not np.isfinite(data_x).all():
        raise ValueError("LSTM features contain nonfinite values")
    fitting = frame.time_utc < cutoff
    scaler_x = RobustScaler().fit(data_x[fitting])
    scaler_y = RobustScaler().fit(data_y[fitting])
    data_x = scaler_x.transform(data_x).astype(np.float32)
    data_y = scaler_y.transform(data_y).astype(np.float32)

    class Batches(keras.utils.Sequence):
        def __init__(self, indices):
            super().__init__()
            self.indices = indices

        def __len__(self):
            return math.ceil(len(self.indices) / spec.batch_size)

        def __getitem__(self, index):
            indices = self.indices[index * spec.batch_size : (index + 1) * spec.batch_size]
            x = np.stack([data_x[i : i + spec.lookback_hours] for i in indices])
            y = np.stack([data_y[i + spec.lookback_hours : i + length, 0] for i in indices])
            return x, y

    model = keras.Sequential([keras.Input(shape=(spec.lookback_hours, len(spec.features)))])
    for i, units in enumerate(spec.units):
        model.add(
            keras.layers.LSTM(units, return_sequences=i < len(spec.units) - 1, dropout=spec.dropout)
        )
    model.add(keras.layers.Dense(spec.dense_units))
    model.add(keras.layers.Dense(spec.forecast_hours))
    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(
        Batches(training),
        validation_data=Batches(validation),
        epochs=spec.epochs,
        shuffle=False,
        verbose=2,
    )
    with atomic_path(artifact(config, domain, "lstm", "keras")) as temporary:
        model.save(temporary)
    with atomic_path(artifact(config, domain, "scalers", "joblib")) as temporary:
        joblib.dump({"x": scaler_x, "y": scaler_y}, temporary)
    write_csv(pd.DataFrame(history.history), artifact(config, domain, "training_history", "csv"))
    metadata = {
        "lstm": spec.model_dump(mode="json"),
        "training": period.model_dump(mode="json"),
        "validation_cutoff": cutoff.isoformat(),
        "training_windows": len(training),
        "validation_windows": len(validation),
        "science": science_contract(config, domain),
    }
    with atomic_path(artifact(config, domain, "lstm_metadata", "json")) as temporary:
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_lstm(config, domain):
    metadata = json.loads(source_artifact(config, domain, "lstm_metadata", "json").read_text())
    if metadata["lstm"] != config.models.lstm.model_dump(mode="json") or metadata[
        "training"
    ] != config.models.lstm_training.model_dump(mode="json"):
        raise ValueError("LSTM artifact configuration mismatch; retrain")
    if metadata["science"] != json.loads(json.dumps(science_contract(config, domain))):
        raise ValueError("LSTM scientific input configuration mismatch; retrain")
    model = keras_backend().models.load_model(
        source_artifact(config, domain, "lstm", "keras"), compile=False
    )
    scalers = joblib.load(source_artifact(config, domain, "scalers", "joblib"))
    return model, scalers


def lstm_predict(model, scalers, data, spec):
    if data.shape[1:] != (spec.lookback_hours, len(spec.features)):
        raise ValueError(f"LSTM input shape mismatch: {data.shape}")
    flat = data.reshape(-1, data.shape[-1])
    scaled = scalers["x"].transform(flat).reshape(data.shape)
    result = np.asarray(model.predict(scaled, batch_size=spec.batch_size, verbose=0))
    if result.shape != (len(data), spec.forecast_hours):
        raise ValueError(f"LSTM output shape mismatch: {result.shape}")
    result = scalers["y"].inverse_transform(result.reshape(-1, 1)).reshape(result.shape)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite inverse-scaled LSTM predictions")
    return result


def site_lstm(config, day, domain):
    spec = config.models.lstm
    model, scalers = load_lstm(config, domain)
    frame = table(config, domain, history_start(config, day), day + timedelta(days=1))
    starts = window_starts(frame, spec.lookback_hours)
    issuance = frame.time_utc.iloc[starts] + pd.Timedelta(hours=spec.lookback_hours)
    keep = (issuance >= pd.Timestamp(day)) & (issuance < pd.Timestamp(day) + pd.Timedelta(days=1))
    starts = starts[keep.to_numpy()]
    if not len(starts):
        raise ValueError(f"No contiguous LSTM station histories for {day}/{domain}")
    records = []
    forecasts = []
    for batch in range(0, len(starts), spec.batch_size):
        indices = starts[batch : batch + spec.batch_size]
        data = np.stack(
            [
                frame.iloc[i : i + spec.lookback_hours][spec.features].to_numpy(dtype=np.float32)
                for i in indices
            ]
        )
        values = lstm_predict(model, scalers, data, spec)
        for index, value in zip(indices, values, strict=True):
            row = frame.iloc[index]
            reference = row.time_utc + pd.Timedelta(hours=spec.lookback_hours)
            for lead, forecast in enumerate(concentration(value, spec)):
                forecasts.append(
                    {
                        "site_index": row.site_index,
                        "lat": row.lat,
                        "lon": row.lon,
                        "forecast_reference_time": reference,
                        "time_utc": reference + pd.Timedelta(hours=lead),
                        "lead_hour": lead,
                        "lstm_pm25": forecast,
                    }
                )
            records.append(
                {
                    "site_index": row.site_index,
                    "time_utc": row.time_utc + pd.Timedelta(hours=spec.lookback_hours),
                    "lstm_predict_log_pm25": value[0],
                    "lstm_pm25": concentration(value[0], spec),
                }
            )
    result = pd.DataFrame(records)
    if spec.target_transform == "identity":
        # XGBoost's historical feature name always denotes log concentration.
        if (result.lstm_pm25 <= 0).any():
            raise ValueError("Identity LSTM predictions must be positive for XGBoost log feature")
        result["lstm_predict_log_pm25"] = np.log(result.lstm_pm25)
    write_csv(result, product(config, "site_lstm", domain=domain, day=day, suffix="csv"))
    write_csv(
        pd.DataFrame(forecasts),
        product(config, "site_lstm_forecast", domain=domain, day=day, suffix="csv"),
    )


def stacked_table(config, domain, period):
    frame = table(config, domain, period.start, period.end)
    predictions = pd.concat(
        [
            pd.read_csv(
                product(config, "site_lstm", domain=domain, day=day, suffix="csv"),
                dtype={"site_index": str},
            )
            for day in period.days()
        ],
        ignore_index=True,
    )
    predictions["time_utc"] = pd.to_datetime(predictions.time_utc)
    result = frame.merge(predictions, on=["site_index", "time_utc"], validate="one_to_one")
    lost = len(frame) - len(result)
    if lost:
        logging.warning("%s rows lack contiguous LSTM histories for stacking", lost)
    if result.empty:
        raise ValueError("No station/hour keys available for XGBoost stacking")
    return result.sort_values("time_utc").reset_index(drop=True)


def xgb_backend():
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError("XGBoost stages require xgboost; install dafcom[ml]") from exc
    return xgboost


def chronological_folds(times, count):
    """Keep every station for a given valid hour on the same side of each split."""
    unique = np.sort(pd.unique(times))
    splitter = TimeSeriesSplit(n_splits=count)
    return [
        (
            np.flatnonzero(np.isin(times, unique[train])),
            np.flatnonzero(np.isin(times, unique[test])),
        )
        for train, test in splitter.split(unique)
    ]


def train_xgb(config, domain):
    spec = config.models.xgboost
    frame = stacked_table(config, domain, config.models.xgb_training)
    x = frame[spec.features]
    y = frame.obs_pm25
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("XGBoost training data contain nonfinite values")
    cpus = config.slurm.stage_resources.get("train_xgb", config.slurm.resources).cpus
    model = xgb_backend().XGBRegressor(**spec.parameters, n_jobs=cpus, random_state=spec.seed)
    if spec.search:
        search = RandomizedSearchCV(
            model,
            spec.search,
            n_iter=spec.search_iterations,
            scoring=spec.scoring,
            cv=chronological_folds(frame.time_utc.to_numpy(), spec.folds),
            random_state=spec.seed,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(x, y)
        model = search.best_estimator_
        write_csv(pd.DataFrame(search.cv_results_), artifact(config, domain, "xgb_search", "csv"))
    else:
        model.fit(x, y)
    with atomic_path(artifact(config, domain, "xgboost", "json")) as temporary:
        model.save_model(temporary)
    with atomic_path(artifact(config, domain, "xgb_metadata", "json")) as temporary:
        temporary.write_text(
            json.dumps(
                {
                    "xgboost": spec.model_dump(mode="json"),
                    "training": config.models.xgb_training.model_dump(mode="json"),
                    "science": science_contract(config, domain),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_xgb(config, domain, stage="site_xgb"):
    metadata = json.loads(source_artifact(config, domain, "xgb_metadata", "json").read_text())
    metadata["xgboost"].setdefault("scoring", "r2")
    if metadata["xgboost"] != config.models.xgboost.model_dump(mode="json") or metadata[
        "training"
    ] != config.models.xgb_training.model_dump(mode="json"):
        raise ValueError("XGBoost artifact configuration mismatch; retrain")
    if metadata["science"] != json.loads(json.dumps(science_contract(config, domain))):
        raise ValueError("XGBoost scientific input configuration mismatch; retrain")
    model = xgb_backend().XGBRegressor()
    model.load_model(source_artifact(config, domain, "xgboost", "json"))
    cpus = config.slurm.stage_resources.get(stage, config.slurm.resources).cpus
    model.set_params(n_jobs=cpus)
    return model


def site_xgb(config, day, domain):
    from .config import Period

    frame = stacked_table(config, domain, Period(start=day, end=day + timedelta(days=1)))
    model = load_xgb(config, domain)
    frame["xgb_predictions_pm25"] = model.predict(frame[config.models.xgboost.features])
    model_feature = config.variables[config.models.model_variable].feature
    frame["ufs_pm25"] = frame[model_feature]
    if not np.isfinite(frame.xgb_predictions_pm25).all():
        raise ValueError("Nonfinite XGBoost station predictions")
    columns = [
        "site_index",
        "time_utc",
        "lat",
        "lon",
        "obs_pm25",
        "xgb_predictions_pm25",
        "lstm_pm25",
        "ufs_pm25",
    ]
    if config.baseline_variable:
        columns.append("bias_correction_pm25")
    write_csv(frame[columns], product(config, "site_xgb", domain=domain, day=day, suffix="csv"))


def grid_predict(config, day, domain, tile):
    spec = config.models.lstm
    model, scalers = load_lstm(config, domain)
    xgb = load_xgb(config, domain, stage="grid_predict")
    grid = config.domains[domain]
    cell_y, cell_x = np.meshgrid(
        np.arange(tile.y0, tile.y1), np.arange(tile.x0, tile.x1), indexing="ij"
    )
    y, x = cell_y.ravel(), cell_x.ravel()
    start = pd.Timestamp(day)
    times = pd.date_range(
        start - pd.Timedelta(hours=spec.lookback_hours), periods=spec.lookback_hours, freq="h"
    )
    lstm = np.empty((spec.forecast_hours, len(y)), dtype=np.float32)
    xgb_result = np.empty(len(y), dtype=np.float32)
    xgb_features = [
        name for name in config.models.xgboost.features if name != "lstm_predict_log_pm25"
    ]
    for offset in range(0, len(y), config.models.prediction_batch_cells):
        stop = min(offset + config.models.prediction_batch_cells, len(y))
        batch_y, batch_x = y[offset:stop], x[offset:stop]
        data = grid_features(config, domain, tile, times, batch_y, batch_x, spec.features)
        prediction = lstm_predict(model, scalers, data, spec)
        lstm[:, offset:stop] = concentration(prediction, spec).T
        current = grid_features(
            config, domain, tile, pd.DatetimeIndex([start]), batch_y, batch_x, xgb_features
        )
        frame = pd.DataFrame(current[:, 0, :], columns=xgb_features)
        if spec.target_transform == "log":
            frame["lstm_predict_log_pm25"] = prediction[:, 0]
        else:
            if (prediction[:, 0] <= 0).any():
                raise ValueError("Identity LSTM predictions must be positive for log feature")
            frame["lstm_predict_log_pm25"] = np.log(prediction[:, 0])
        xgb_result[offset:stop] = xgb.predict(frame[config.models.xgboost.features])
    if not np.isfinite(xgb_result).all():
        raise ValueError("Nonfinite gridded XGBoost predictions")
    shape = cell_y.shape
    dataset = xr.Dataset(
        {
            "lstm_pm25": (("time", "lat", "lon"), lstm.reshape(spec.forecast_hours, *shape)),
            "xgb_predictions_pm25": (("lat", "lon"), xgb_result.reshape(shape)),
        },
        coords={
            "time": pd.date_range(start, periods=spec.forecast_hours, freq="h"),
            "lat": grid.north - np.arange(tile.y0, tile.y1) * grid.resolution,
            "lon": grid.west + np.arange(tile.x0, tile.x1) * grid.resolution,
        },
        attrs={
            "domain": domain,
            "tile": tile.name,
            "forecast_reference_time": start.isoformat(),
            "xgb_valid_time": start.isoformat(),
            "Conventions": "CF-1.8",
        },
    )
    for name in dataset.data_vars:
        dataset[name].attrs["units"] = config.variables[config.models.model_variable].units
    dataset.lat.attrs["units"] = "degrees_north"
    dataset.lon.attrs["units"] = "degrees_east"
    write_netcdf(dataset, product(config, "grid_predict", domain=domain, day=day, tile=tile.name))
