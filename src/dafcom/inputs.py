"""UFS ingestion, surface extraction, and reusable tiled regridding."""

from __future__ import annotations

import glob
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from .config import render
from .grid import target_grid
from .storage import product, write_csv, write_netcdf


def source_paths(config, variable, day):
    spec = config.variables[variable]
    paths = []
    for offset in spec.cycle_day_offsets:
        cycle_day = day + timedelta(days=offset)
        for lead in spec.leads if spec.file_per_hour else [0]:
            pattern = render(spec.source, config, cycle_day, variable=variable, lead=lead)
            matches = sorted(glob.glob(pattern))
            if not matches or (spec.file_per_hour and len(matches) != 1):
                raise ValueError(
                    f"Expected {'one' if spec.file_per_hour else 'at least one'} "
                    f"source file for {pattern}, found {len(matches)}"
                )
            paths.extend(matches)
    if len(set(paths)) != len(paths):
        raise ValueError(f"Overlapping source patterns for {variable} on {day}")
    return paths


def normalize(dataset, spec, variable):
    dataset = dataset.isel(spec.indexers)
    arrays = [dataset[field] for field in spec.fields]
    if any(array.dims != arrays[0].dims or array.shape != arrays[0].shape for array in arrays):
        raise ValueError(f"Summed fields for {variable} must have identical dimensions")
    data = arrays[0]
    for array in arrays[1:]:
        data = data + array
    lat, lon = dataset[spec.lat], dataset[spec.lon]
    if spec.time in lat.dims:
        first_lat, first_lon = lat.isel({spec.time: 0}), lon.isel({spec.time: 0})
        if not (lat == first_lat).all().item() or not (lon == first_lon).all().item():
            raise ValueError("Time-varying UFS grids require a dedicated adapter")
        lat, lon = first_lat, first_lon
    if lat.ndim == lon.ndim == 1:
        spatial = (lat.dims[0], lon.dims[0])
        latitude = xr.DataArray(lat.values, dims="y")
        longitude = xr.DataArray((lon.values + 180) % 360 - 180, dims="x")
    elif lat.ndim == lon.ndim == 2 and lat.dims == lon.dims:
        spatial = lat.dims
        latitude = xr.DataArray(lat.values, dims=("y", "x"))
        longitude = xr.DataArray((lon.values + 180) % 360 - 180, dims=("y", "x"))
    else:
        raise ValueError(f"Unsupported spatial coordinates for {variable}")
    if len(set(spatial)) != 2:
        raise ValueError("Rectilinear coordinates must have distinct dimensions")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError(f"{variable}: source coordinates must be finite")
    if (np.abs(latitude) > 90).any():
        raise ValueError(f"{variable}: source latitude outside [-90, 90]")
    for dim in tuple(data.dims):
        if dim not in (*spatial, spec.time):
            if data.sizes[dim] != 1:
                raise ValueError(f"Configure indexers for nonsingleton surface dimension {dim}")
            data = data.isel({dim: 0})
    if spec.time in data.dims:
        if spec.kind == "static":
            if data.sizes[spec.time] != 1:
                raise ValueError("Static time dimension must be selected explicitly")
            data = data.isel({spec.time: 0})
        else:
            if spec.time_slice:
                data = data.isel({spec.time: slice(*spec.time_slice)})
            data = data.transpose(spec.time, *spatial)
    if spec.time not in data.dims:
        data = data.transpose(*spatial)
        if spec.kind == "dynamic":
            if spec.time in data.coords and data[spec.time].ndim == 0:
                data = data.expand_dims(spec.time)
            else:
                data = data.expand_dims({spec.time: [0]})
    dims = ("time", "y", "x") if spec.kind == "dynamic" else ("y", "x")
    result = xr.Dataset({variable: (dims, data.values)}, coords={"lat": latitude, "lon": longitude})
    result[variable].attrs = dict(arrays[0].attrs)
    if spec.units:
        result[variable].attrs["units"] = spec.units
    if spec.kind == "dynamic" and spec.time_alignment == "source":
        if spec.time not in data.coords or not np.issubdtype(data[spec.time].dtype, np.datetime64):
            raise ValueError(f"{variable}: source requires decoded valid-time coordinates")
        result = result.assign_coords(time=data[spec.time].values)
    return result


def ingest(config, day, variable):
    ingest_many(config, day, [variable])


def ingest_many(config, day, variables):
    groups = {}
    for variable in variables:
        if config.variables[variable].kind != "dynamic":
            raise ValueError("ingest is only for dynamic variables")
        paths = tuple(source_paths(config, variable, day))
        groups.setdefault(paths, []).append(variable)
    for paths, names in groups.items():
        frames = {name: [] for name in names}
        references = {}
        for path in paths:
            with xr.open_dataset(path) as dataset:
                for name in names:
                    frame = normalize(dataset, config.variables[name], name)
                    reference = references.get(name)
                    if reference is not None and (
                        not frame.lat.equals(reference.lat) or not frame.lon.equals(reference.lon)
                    ):
                        raise ValueError(f"{name}: source grid changes across forecast files")
                    references[name] = frame
                    frames[name].append(frame)
        for name in names:
            write_daily(config, day, name, frames[name], paths)


def write_daily(config, day, variable, frames, paths):
    spec = config.variables[variable]
    data = xr.concat(frames, dim="time", data_vars="all", coords="minimal", compat="equals")
    start = pd.Timestamp(day)
    expected = pd.date_range(start, periods=24, freq="h")
    if spec.time_alignment == "legacy_index":
        data = data.assign_coords(
            time=pd.date_range(
                start + pd.Timedelta(hours=spec.legacy_hour_offset),
                periods=data.sizes["time"],
                freq="h",
            )
        )
    data = data.sel(time=(data.time >= start) & (data.time < start + pd.Timedelta(days=1)))
    data = data.sortby("time")
    if not np.array_equal(data.time.values, expected.values):
        raise ValueError(
            f"{variable} on {day}: require exactly 24 unique hourly valid times, "
            f"found {data.sizes['time']}"
        )
    data.attrs.update(source_files=";".join(paths), time_alignment=spec.time_alignment)
    write_netcdf(data, product(config, "native", day=day, variable=variable))


def regrid_data(source, target, variable, method, regridders=None):
    lat, lon = source.lat, source.lon
    if lat.ndim == lon.ndim == 2:
        if np.allclose(lat, lat.values[:, :1], rtol=0, atol=1e-8) and np.allclose(
            lon, lon.values[:1, :], rtol=0, atol=1e-8
        ):
            lat = lat.isel(x=0)
            lon = lon.isel(y=0)
    if lat.ndim == lon.ndim == 1:
        regular = (
            source[variable].drop_vars(["lat", "lon"]).assign_coords(y=lat.values, x=lon.values)
        )
        regular = regular.rename({"y": "lat", "x": "lon"}).sortby("lat").sortby("lon")
        for coord in ("lat", "lon"):
            if np.any(np.diff(regular[coord].values) <= 0):
                raise ValueError(f"Source {coord} must be unique")
        result = regular.interp(
            lat=target.lat, lon=target.lon, method="linear" if method == "bilinear" else "nearest"
        )
    else:
        try:
            import xesmf as xe
        except ImportError as exc:
            raise RuntimeError(
                "Curvilinear UFS regridding requires xESMF and ESMF/esmpy; "
                "install the regrid extra in the Ursa ESMF environment"
            ) from exc
        key = hashlib.sha256()
        for coordinate in (source.lat, source.lon, target.lat, target.lon):
            key.update(str((coordinate.shape, coordinate.dtype)).encode())
            key.update(coordinate.values.tobytes())
        key.update(method.encode())
        cache = {} if regridders is None else regridders
        signature = key.hexdigest()
        if signature not in cache:
            with TemporaryDirectory(prefix="dafcom-weights-") as directory:
                cache[signature] = xe.Regridder(
                    source,
                    target,
                    method,
                    filename=str(Path(directory) / "weights.nc"),
                    unmapped_to_nan=True,
                )
        result = cache[signature](source[variable], keep_attrs=True).load()
    result.name = variable
    return result.to_dataset()


def regrid(config, day, domain, tile, variable, regridders=None):
    spec = config.variables[variable]
    if spec.kind == "static":
        paths = source_paths(config, variable, day)
        if len(paths) != 1:
            raise ValueError(f"Static variable {variable} must resolve to one file")
        with xr.open_dataset(paths[0]) as dataset:
            source = normalize(dataset, spec, variable)
        stage = "static"
    else:
        with xr.open_dataset(product(config, "native", day=day, variable=variable)) as dataset:
            source = dataset.load()
        stage = "regrid"
    result = regrid_data(
        source, target_grid(config.domains[domain], tile), variable, spec.regrid_method, regridders
    )
    result.attrs.update(
        source_files=source.attrs.get("source_files", spec.source),
        domain=domain,
        tile=tile.name,
        regrid_method=spec.regrid_method,
    )
    write_netcdf(
        result,
        product(
            config,
            stage,
            domain=domain,
            day=day
            if stage == "regrid"
            else day.replace(day=1)
            if spec.static_period == "monthly"
            else "",
            tile=tile.name,
            variable=variable,
        ),
    )


def observations(config, day, domain):
    spec = config.observations
    path = render(spec.source, config, day)
    if spec.format == "netcdf":
        with xr.open_dataset(path) as source:
            source = source.isel(spec.indexers)
            data = source[spec.variable]
            for dim in tuple(data.dims):
                if dim != spec.time and dim not in source[spec.lat].dims:
                    if data.sizes[dim] != 1:
                        raise ValueError(f"Configure observation indexers for dimension {dim}")
                    data = data.isel({dim: 0})
            for sentinel in spec.missing_values:
                data = data.where(data != sentinel)
            data = data.resample({spec.time: "1h"}).mean()
            station_dim = source[spec.lat].dims
            if len(station_dim) != 1:
                raise ValueError("Observation latitude must be a one-dimensional station array")
            station_dim = station_dim[0]
            sites = source[spec.site].values if spec.site else np.arange(source.sizes[station_dim])
            frame = (
                data.assign_coords(
                    {
                        "lat": (station_dim, source[spec.lat].values),
                        "lon": (station_dim, (source[spec.lon].values + 180) % 360 - 180),
                        "site_index": (station_dim, sites),
                    }
                )
                .to_dataframe(name="obs_pm25")
                .reset_index()
            )
            frame = frame.rename(columns={spec.time: "time_utc"})
    else:
        key = spec.site or "site_index"
        frame = (
            pd.read_csv(path, dtype={key: str})
            if spec.format == "csv"
            else pd.read_excel(path, dtype={key: str})
        )
        frame = frame.rename(
            columns={
                spec.time: "time_utc",
                spec.lat: "lat",
                spec.lon: "lon",
                spec.variable: "obs_pm25",
                spec.site or "site_index": "site_index",
            }
        )
        for sentinel in spec.missing_values:
            frame.loc[frame.obs_pm25 == sentinel, "obs_pm25"] = np.nan
        frame["time_utc"] = pd.to_datetime(frame.time_utc, utc=True).dt.tz_localize(None)
        if frame.groupby("site_index")[["lat", "lon"]].nunique().gt(1).any().any():
            raise ValueError("Station coordinates change within observation input")
        frame = frame.groupby(["site_index", pd.Grouper(key="time_utc", freq="h")], as_index=False)[
            ["lat", "lon", "obs_pm25"]
        ].mean()
        frame["lon"] = (frame.lon + 180) % 360 - 180
    frame = frame[["site_index", "time_utc", "lat", "lon", "obs_pm25"]].copy()
    if frame.site_index.isna().any():
        raise ValueError("Observation station IDs cannot be missing")
    frame["time_utc"] = pd.to_datetime(frame.time_utc, utc=True).dt.tz_localize(None)
    start = datetime.combine(day, datetime.min.time())
    grid = config.domains[domain]
    keep = (
        frame.time_utc.ge(start)
        & frame.time_utc.lt(start + timedelta(days=1))
        & frame.obs_pm25.gt(spec.minimum)
        & np.isfinite(frame.obs_pm25)
        & frame.lat.between(grid.south, grid.north)
        & frame.lon.between(grid.west, grid.east)
    )
    frame = frame.loc[keep].sort_values(["site_index", "time_utc"])
    if frame.duplicated(["site_index", "time_utc"]).any():
        raise ValueError("Observation station/hour keys must be unique")
    if spec.round_digits is not None:
        frame["obs_pm25"] = frame.obs_pm25.round(spec.round_digits)
    write_csv(frame, product(config, "observations", domain=domain, day=day, suffix="csv"))
