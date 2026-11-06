"""Vectorized station sampling and bounded-memory gridded feature loading."""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

from .grid import distances, owners, tiles
from .storage import product, write_csv

DISTANCE_FEATURES = ["v22_d1", "v23_d2", "v24_d3", "v24_d4", "v25_d5"]
DERIVED_FEATURES = ["v20_lon", "v21_lat", *DISTANCE_FEATURES, "hour_utc", "day_of_year"]


def field_path(config, variable, domain, day, tile):
    spec = config.variables[variable]
    static = spec.kind == "static"
    return product(
        config,
        "static" if static else "regrid",
        domain=domain,
        day=day.replace(day=1)
        if static and spec.static_period == "monthly"
        else ""
        if static
        else day,
        tile=tile.name,
        variable=variable,
    )


def sample(config, day, domain):
    frame = pd.read_csv(
        product(config, "observations", domain=domain, day=day, suffix="csv"),
        dtype={"site_index": str},
    )
    frame["time_utc"] = pd.to_datetime(frame.time_utc)
    valid, y, x = owners(config.domains[domain], frame.lat.to_numpy(), frame.lon.to_numpy())
    if (~valid).any():
        logging.warning(
            "Excluding %s stations outside the half-open grid coordinates", (~valid).sum()
        )
    frame = frame.loc[valid].copy()
    y, x = y[valid], x[valid]
    for tile in tiles(config.domains[domain]):
        mask = (y >= tile.y0) & (y < tile.y1) & (x >= tile.x0) & (x < tile.x1)
        points = frame.loc[mask]
        if points.empty:
            continue
        latitude = xr.DataArray(points.lat.to_numpy(), dims="point")
        longitude = xr.DataArray(points.lon.to_numpy(), dims="point")
        time = xr.DataArray(points.time_utc.to_numpy(), dims="point")
        for variable, spec in config.variables.items():
            if spec.feature is None and variable != config.baseline_variable:
                continue
            with xr.open_dataset(field_path(config, variable, domain, day, tile)) as dataset:
                data = dataset[variable]
                spatial = data.interp(lat=latitude, lon=longitude, method="linear")
                if spec.kind == "dynamic":
                    spatial = spatial.sel(time=time)
                values = spatial.to_numpy()
            name = spec.feature or "bias_correction_pm25"
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Nonfinite {variable} station samples on {day}/{domain}/{tile.name}"
                )
            frame.loc[mask, name] = values
            if variable == config.baseline_variable:
                frame.loc[mask, "bias_correction_pm25"] = values
    frame["v20_lon"] = frame.lon
    frame["v21_lat"] = frame.lat
    for name, values in zip(
        DISTANCE_FEATURES,
        distances(frame.lat.to_numpy(), frame.lon.to_numpy(), config.distance_reference),
        strict=True,
    ):
        frame[name] = values
    frame["hour_utc"] = frame.time_utc.dt.hour
    frame["day_of_year"] = frame.time_utc.dt.day_of_year
    frame["log_pm25"] = np.log(frame.obs_pm25) if len(frame) else pd.Series(dtype=float)
    for spec in config.variables.values():
        if spec.feature and spec.feature not in frame:
            frame[spec.feature] = pd.Series(dtype=float)
    if frame.empty:
        logging.warning("No usable station observations on %s/%s", day, domain)
    write_csv(frame, product(config, "features", domain=domain, day=day, suffix="csv"))


def table(config, domain, start, end):
    days = pd.date_range(start, end, freq="D", inclusive="left")
    frames = [
        pd.read_csv(
            product(config, "features", domain=domain, day=day.date(), suffix="csv"),
            dtype={"site_index": str},
        )
        for day in days
    ]
    if not frames:
        raise ValueError("Feature table period is empty")
    frame = pd.concat(frames, ignore_index=True)
    frame["time_utc"] = pd.to_datetime(frame.time_utc)
    if frame.duplicated(["site_index", "time_utc"]).any():
        raise ValueError("Duplicate feature station/hour keys")
    if frame.groupby("site_index")[["lat", "lon"]].nunique().gt(1).any().any():
        raise ValueError(
            "Station identity/coordinates changed; configure a stable observation site ID"
        )
    return frame.sort_values(["site_index", "time_utc"]).reset_index(drop=True)


def grid_features(config, domain, tile, times, cell_y, cell_x, names):
    """Return (cells, hours, features); static fields are read only once per batch."""
    output = np.empty((len(cell_y), len(times), len(names)), dtype=np.float32)
    grid = config.domains[domain]
    latitude = grid.north - cell_y * grid.resolution
    longitude = grid.west + cell_x * grid.resolution
    derived = {
        "v20_lon": np.broadcast_to(longitude[:, None], output.shape[:2]),
        "v21_lat": np.broadcast_to(latitude[:, None], output.shape[:2]),
        "hour_utc": np.broadcast_to(times.hour, output.shape[:2]),
        "day_of_year": np.broadcast_to(times.day_of_year, output.shape[:2]),
    }
    for name, value in zip(
        DISTANCE_FEATURES, distances(latitude, longitude, config.distance_reference), strict=True
    ):
        derived[name] = np.broadcast_to(value[:, None], output.shape[:2])
    specs = {spec.feature: variable for variable, spec in config.variables.items() if spec.feature}
    for i, name in enumerate(names):
        if name in derived:
            output[:, :, i] = derived[name]
            continue
        variable = specs[name]
        spec = config.variables[variable]
        values = np.empty(output.shape[:2], dtype=np.float32)
        for day in sorted(set(times.date)):
            with xr.open_dataset(field_path(config, variable, domain, day, tile)) as dataset:
                data = dataset[variable].sel(
                    lat=xr.DataArray(latitude, dims="cell"),
                    lon=xr.DataArray(longitude, dims="cell"),
                    method="nearest",
                    tolerance=grid.resolution / 100,
                )
                if spec.kind == "static":
                    if spec.static_period == "fixed":
                        values[:] = data.values[:, None]
                        break
                    mask = times.date == day
                    values[:, mask] = data.values[:, None]
                    continue
                mask = times.date == day
                data = data.sel(time=times[mask].values).transpose("cell", "time")
                values[:, mask] = data.values
        output[:, :, i] = values
    if not np.isfinite(output).all():
        raise ValueError(f"Nonfinite gridded features for {domain}/{tile.name}")
    return output


def history_start(config, day):
    return (pd.Timestamp(day) - timedelta(hours=config.models.lstm.lookback_hours)).date()
