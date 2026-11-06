import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dafcom.config import Config


@pytest.fixture
def config(tmp_path):
    return Config.model_validate(
        {
            "run": {"start": "2026-01-01", "end": "2026-01-13"},
            "output_root": tmp_path / "output",
            "domains": {
                "tiny": {
                    "south": 0,
                    "north": 2,
                    "west": 0,
                    "east": 2,
                    "resolution": 0.5,
                    "tile_shape": [2, 2],
                },
            },
            "variables": {
                "pm25": {
                    "source": str(tmp_path / "raw_{date}.nc"),
                    "fields": ["pm25"],
                    "feature": "v13_ufs_pm25",
                    "units": "ug/m3",
                },
                "elevation": {
                    "source": str(tmp_path / "elevation.nc"),
                    "fields": ["elevation"],
                    "kind": "static",
                    "feature": "v11_elevation",
                    "units": "m",
                },
            },
            "observations": {
                "source": str(tmp_path / "obs_{date}.nc"),
                "variable": "PM2.5",
                "site": "station_id",
            },
            "models": {
                "lstm_training": {"start": "2026-01-01", "end": "2026-01-09"},
                "xgb_training": {"start": "2026-01-09", "end": "2026-01-12"},
                "prediction": {"start": "2026-01-12", "end": "2026-01-13"},
                "lstm": {
                    "features": ["v13_ufs_pm25", "v11_elevation", "hour_utc", "day_of_year"],
                    "lookback_hours": 3,
                    "forecast_hours": 2,
                    "units": [4],
                    "dense_units": 4,
                    "epochs": 1,
                    "batch_size": 16,
                },
                "xgboost": {
                    "features": ["v13_ufs_pm25", "v11_elevation", "lstm_predict_log_pm25"],
                    "parameters": {"n_estimators": 4, "max_depth": 2},
                    "folds": 2,
                },
                "prediction_batch_cells": 3,
            },
            "slurm": {
                "resources": {"cpus": 1, "memory": "1G", "walltime": "00:10:00"},
                "max_array_size": 10,
                "max_concurrent": 2,
            },
        }
    )


@pytest.fixture
def synthetic_data(config):
    latitude = np.linspace(2, 0, 5)
    longitude = np.linspace(0, 2, 5)
    elevation = 2 * latitude[:, None] + 3 * longitude[None, :]
    xr.Dataset(
        {"elevation": (("lat", "lon"), elevation)}, coords={"lat": latitude, "lon": longitude}
    ).to_netcdf(config.variables["elevation"].source)
    for day in config.run.days():
        times = pd.date_range(day, periods=24, freq="h")
        offset = (day - config.run.start).days * 24
        hour = np.arange(24) + offset
        values = 10 + np.sin(hour[:, None, None] / 24) + elevation[None, :, :]
        xr.Dataset(
            {"pm25": (("time", "lat", "lon"), values)},
            coords={"time": times, "lat": latitude, "lon": longitude},
        ).to_netcdf(config.variables["pm25"].source.format(date=day.strftime("%Y%m%d")))
        station_lat, station_lon = [1.25, 0.5], [0.75, 1.5]
        observed = (
            12
            + np.sin(hour[:, None] / 24)
            + (2 * np.array(station_lat) + 3 * np.array(station_lon))[None, :]
        )
        xr.Dataset(
            {
                "PM2.5": (("time", "station"), observed),
                "latitude": ("station", station_lat),
                "longitude": ("station", station_lon),
                "station_id": ("station", [101, 202]),
            },
            coords={"time": times, "station": [0, 1]},
        ).to_netcdf(config.observations.source.format(date=day.strftime("%Y%m%d")))
    return config
