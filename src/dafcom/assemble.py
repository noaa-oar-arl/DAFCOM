"""Stream forecast tiles into a CF domain product without full-domain RAM."""

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, date2num

from .grid import coordinates, tiles
from .storage import atomic_path, product


def assemble(config, day, domain):
    grid = config.domains[domain]
    latitude, longitude = coordinates(grid)
    times = pd.date_range(day, periods=config.models.lstm.forecast_hours, freq="h")
    output = product(config, "forecast", domain=domain, day=day)
    units = config.variables[config.models.model_variable].units
    with atomic_path(output) as temporary:
        with Dataset(temporary, "w", format="NETCDF4") as target:
            target.createDimension("time", len(times))
            target.createDimension("lat", len(latitude))
            target.createDimension("lon", len(longitude))
            lat = target.createVariable("lat", "f8", ("lat",))
            lon = target.createVariable("lon", "f8", ("lon",))
            time = target.createVariable("time", "f8", ("time",))
            lat.units, lon.units = "degrees_north", "degrees_east"
            time.units, time.calendar = "hours since 1970-01-01 00:00:00", "standard"
            time[:] = date2num(times.to_pydatetime(), time.units, time.calendar)
            lat[:], lon[:] = latitude, longitude
            lstm = target.createVariable(
                "lstm_pm25",
                "f4",
                ("time", "lat", "lon"),
                zlib=True,
                complevel=1,
                chunksizes=(
                    1,
                    min(grid.tile_shape[0], len(latitude)),
                    min(grid.tile_shape[1], len(longitude)),
                ),
            )
            xgb = target.createVariable(
                "xgb_predictions_pm25", "f4", ("lat", "lon"), zlib=True, complevel=1
            )
            lstm.units = xgb.units = units
            target.Conventions = "CF-1.8"
            target.domain = domain
            target.forecast_reference_time = pd.Timestamp(day).isoformat()
            xgb.valid_time = pd.Timestamp(day).isoformat()
            for tile in tiles(grid):
                path = product(config, "grid_predict", domain=domain, day=day, tile=tile.name)
                ys, xs = tile.slices(grid)
                with xr.open_dataset(path) as source:
                    if not (
                        np.array_equal(source.lat, latitude[ys])
                        and np.array_equal(source.lon, longitude[xs])
                        and np.array_equal(source.time.values, times.values)
                    ):
                        raise ValueError(f"Forecast tile coordinates/time mismatch: {path}")
                    expected = (len(times), tile.y1 - tile.y0, tile.x1 - tile.x0)
                    if source.lstm_pm25.shape != expected:
                        raise ValueError(f"Forecast tile shape mismatch: {path}")
                    if source.xgb_predictions_pm25.shape != expected[1:]:
                        raise ValueError(f"XGBoost forecast tile shape mismatch: {path}")
                    lstm_values = source.lstm_pm25.values
                    xgb_values = source.xgb_predictions_pm25.values
                    if not np.isfinite(lstm_values).all() or not np.isfinite(xgb_values).all():
                        raise ValueError(f"Nonfinite forecast tile: {path}")
                    lstm[:, ys, xs] = lstm_values
                    xgb[ys, xs] = xgb_values
