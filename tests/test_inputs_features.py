from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dafcom import features, inputs
from dafcom.config import Variable
from dafcom.grid import tiles
from dafcom.storage import atomic_path, product
from dafcom.workflow import local_run, prepare


def test_analytic_regrid_and_station_sampling(synthetic_data, tmp_path):
    config = synthetic_data
    run_dir = prepare(config, tmp_path / "workflow", ["features"])
    local_run(run_dir)
    from dafcom.config import load_config

    config = load_config(run_dir / "config.yaml")
    frame = pd.read_csv(
        product(config, "features", domain="tiny", day=config.run.start, suffix="csv")
    )
    assert len(frame) == 48
    assert not frame.duplicated(["site_index", "time_utc"]).any()
    expected = (
        10 + 2 * frame.lat + 3 * frame.lon + np.sin(pd.to_datetime(frame.time_utc).dt.hour / 24)
    )
    np.testing.assert_allclose(frame.v13_ufs_pm25, expected, atol=1e-6)
    np.testing.assert_allclose(frame.v11_elevation, 2 * frame.lat + 3 * frame.lon)
    assert frame.site_index.nunique() == 2
    tile = tiles(config.domains["tiny"])[0]
    times = pd.date_range(config.run.start, periods=3, freq="h")
    batch = features.grid_features(
        config,
        "tiny",
        tile,
        times,
        np.array([0, 1]),
        np.array([0, 1]),
        config.models.lstm.features,
    )
    assert batch.shape == (2, 3, 4)
    np.testing.assert_allclose(batch[0, :, 0], 14 + np.sin(np.arange(3) / 24), atol=1e-6)


def test_ingest_missing_duplicate_and_invalid_time(synthetic_data):
    config = synthetic_data
    day = config.run.start
    path = config.variables["pm25"].source.format(date=day.strftime("%Y%m%d"))
    with xr.open_dataset(path) as source:
        dataset = source.load()
    dataset.isel(time=slice(0, 23)).to_netcdf(path)
    with pytest.raises(ValueError, match="24 unique"):
        inputs.ingest(config, day, "pm25")
    dataset.assign_coords(time=np.zeros(24)).to_netcdf(path)
    with pytest.raises(ValueError, match="valid-time"):
        inputs.ingest(config, day, "pm25")
    duplicated = dataset.time.values.copy()
    duplicated[-1] = duplicated[0]
    dataset.assign_coords(time=duplicated).to_netcdf(path)
    with pytest.raises(ValueError, match="24 unique"):
        inputs.ingest(config, day, "pm25")


def test_legacy_timing_is_explicit(synthetic_data):
    config = synthetic_data
    day = config.run.start
    config.variables["pm25"].time_alignment = "legacy_index"
    path = config.variables["pm25"].source.format(date=day.strftime("%Y%m%d"))
    with xr.open_dataset(path) as source:
        dataset = source.load()
    dataset.assign_coords(time=np.arange(24)).to_netcdf(path)
    inputs.ingest(config, day, "pm25")
    with xr.open_dataset(product(config, "native", day=day, variable="pm25")) as result:
        assert result.attrs["time_alignment"] == "legacy_index"
        assert result.time.values[0] == np.datetime64(day)


def test_surface_sum_and_longitudes():
    source = xr.Dataset(
        {
            "a": (("time", "z", "y", "x"), np.ones((1, 2, 2, 2))),
            "b": (("time", "z", "y", "x"), np.ones((1, 2, 2, 2))),
        },
        coords={
            "time": [np.datetime64("2026-01-01")],
            "lat": ("y", [1, 0]),
            "lon": ("x", [235, 236]),
        },
    )
    spec = Variable(source="none", fields=["a", "b"])
    with pytest.raises(ValueError, match="indexers"):
        inputs.normalize(source, spec, "sum")
    spec.indexers = {"z": 0}
    result = inputs.normalize(source, spec, "sum")
    np.testing.assert_array_equal(result["sum"], 2)
    np.testing.assert_array_equal(result.lon, [-125, -124])


def test_observation_sentinel_masked_before_hourly_mean(config):
    day = config.run.start
    path = config.observations.source.format(date=day.strftime("%Y%m%d"))
    xr.Dataset(
        {
            "PM2.5": (("time", "station"), [[-1], [10]]),
            "latitude": ("station", [1]),
            "longitude": ("station", [1]),
            "station_id": ("station", [8]),
        },
        coords={"time": pd.date_range(day, periods=2, freq="30min"), "station": [0]},
    ).to_netcdf(path)
    inputs.observations(config, day, "tiny")
    frame = pd.read_csv(product(config, "observations", domain="tiny", day=day, suffix="csv"))
    assert frame.obs_pm25.tolist() == [10]


def test_atomic_failure_leaves_old_file(tmp_path):
    path = tmp_path / "result.csv"
    path.write_text("old")
    with pytest.raises(RuntimeError):
        with atomic_path(path) as temporary:
            temporary.write_text("new")
            raise RuntimeError("interrupted")
    assert path.read_text() == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_monthly_static_path(config):
    spec = config.variables["elevation"]
    spec.static_period = "monthly"
    tile = tiles(config.domains["tiny"])[0]
    day = config.run.start + timedelta(days=3)
    assert "/2026-01-01/" in str(features.field_path(config, "elevation", "tiny", day, tile))


def test_monthly_fields_change_across_month_boundary(config, tmp_path):
    from datetime import date

    spec = config.variables["elevation"]
    spec.static_period = "monthly"
    spec.source = str(tmp_path / "static_{month}.nc")
    tile = tiles(config.domains["tiny"])[0]
    for day, value in [(date(2026, 1, 1), 10), (date(2026, 2, 1), 20)]:
        xr.Dataset(
            {"elevation": (("lat", "lon"), np.full((5, 5), value, dtype=float))},
            coords={"lat": np.linspace(2, 0, 5), "lon": np.linspace(0, 2, 5)},
        ).to_netcdf(spec.source.format(month=day.strftime("%m")))
        inputs.regrid(config, day, "tiny", tile, "elevation")
    result = features.grid_features(
        config,
        "tiny",
        tile,
        pd.date_range("2026-01-31T23", periods=2, freq="h"),
        np.array([0]),
        np.array([0]),
        ["v11_elevation"],
    )
    np.testing.assert_array_equal(result[0, :, 0], [10, 20])


def test_shared_input_files_open_once(synthetic_data, monkeypatch):
    config = synthetic_data
    config.variables["copy"] = config.variables["pm25"].model_copy()
    opened = []
    original = inputs.xr.open_dataset

    def counted(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(inputs.xr, "open_dataset", counted)
    inputs.ingest_many(config, config.run.start, ["pm25", "copy"])
    assert len(opened) == 1


def test_scalar_valid_time():
    source = xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2)))},
        coords={"time": np.datetime64("2026-01-01T04"), "lat": ("y", [1, 0]), "lon": ("x", [0, 1])},
    )
    result = inputs.normalize(source, Variable(source="unused", fields=["a"]), "a")
    assert result.time.values[0] == np.datetime64("2026-01-01T04")


def test_file_per_hour_previous_cycle_crosses_year(config, tmp_path):
    day = config.run.start
    spec = config.variables["pm25"]
    spec.source = str(tmp_path / "cycle_{date}_{lead:03d}.nc")
    spec.file_per_hour = True
    spec.leads = list(range(1, 25))
    spec.cycle_day_offsets = [-1, 0]
    for cycle_day in [day - timedelta(days=1), day]:
        for lead in spec.leads:
            time = pd.Timestamp(cycle_day) + pd.Timedelta(hours=12 + lead)
            xr.Dataset(
                {"pm25": (("time", "lat", "lon"), np.full((1, 2, 2), time.hour, dtype=float))},
                coords={"time": [time], "lat": [2, 0], "lon": [0, 2]},
            ).to_netcdf(spec.source.format(date=cycle_day.strftime("%Y%m%d"), lead=lead))
    inputs.ingest(config, day, "pm25")
    with xr.open_dataset(product(config, "native", day=day, variable="pm25")) as result:
        np.testing.assert_array_equal(result.pm25[:, 0, 0], np.arange(24))


def test_csv_observations_preserve_station_identity(config, tmp_path):
    spec = config.observations
    spec.format = "csv"
    spec.source = str(tmp_path / "observations.csv")
    spec.site, spec.time, spec.lat, spec.lon, spec.variable = (
        "station",
        "valid",
        "latitude",
        "longitude",
        "observed",
    )
    pd.DataFrame(
        {
            "station": ["001", "001", "1", "1"],
            "valid": ["2026-01-01T00:00", "2026-01-01T00:30"] * 2,
            "latitude": [1.25, 1.25, 0.5, 0.5],
            "longitude": [0.75, 0.75, 1.5, 1.5],
            "observed": [10, -1, 20, 20],
        }
    ).to_csv(spec.source, index=False)
    inputs.observations(config, config.run.start, "tiny")
    result = pd.read_csv(
        product(config, "observations", domain="tiny", day=config.run.start, suffix="csv"),
        dtype={"site_index": str},
    )
    assert result.site_index.tolist() == ["001", "1"]
    assert result.obs_pm25.tolist() == [10, 20]


def test_curvilinear_backend_reuses_weights(config, monkeypatch):
    import sys
    from types import SimpleNamespace

    source = xr.Dataset(
        {"a": (("y", "x"), np.ones((2, 2))), "b": (("y", "x"), np.ones((2, 2)))},
        coords={
            "lat": (("y", "x"), [[2, 2.1], [1, 1.1]]),
            "lon": (("y", "x"), [[0, 1], [0.1, 1.1]]),
        },
    )
    target = xr.Dataset(coords={"lat": [1.5, 1], "lon": [0.25, 0.75]})
    calls = []

    class Regridder:
        def __init__(self, src, dst, method, **kwargs):
            calls.append(kwargs)

        def __call__(self, data, **kwargs):
            return xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"), coords=target.coords)

    monkeypatch.setitem(sys.modules, "xesmf", SimpleNamespace(Regridder=Regridder))
    cache = {}
    inputs.regrid_data(source, target, "a", "bilinear", cache)
    inputs.regrid_data(source, target, "b", "bilinear", cache)
    assert len(calls) == 1
    assert calls[0]["unmapped_to_nan"]


def test_subdomains_share_native_inputs_but_not_products(synthetic_data, tmp_path):
    from dafcom.config import Grid, load_config

    config = synthetic_data
    config.domains["east"] = Grid(
        south=0, north=2, west=1, east=2, resolution=0.5, tile_shape=(2, 2)
    )
    directory = prepare(config, tmp_path / "subdomains", ["features"])
    local_run(directory, workers=2)
    config = load_config(directory / "config.yaml")
    assert len(list((config.output_root / "native").glob("*/*.nc"))) == 12
    tiny = pd.read_csv(
        product(config, "features", domain="tiny", day=config.run.start, suffix="csv")
    )
    east = pd.read_csv(
        product(config, "features", domain="east", day=config.run.start, suffix="csv")
    )
    assert len(tiny) == 48
    assert len(east) == 24
    assert east.site_index.unique().tolist() == [202]
