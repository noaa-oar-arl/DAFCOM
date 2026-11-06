from datetime import date
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dafcom.config import Config, Grid, Period, load_config, render, save_config
from dafcom.grid import coordinates, distances, owners, target_grid, tiles
from dafcom.workflow import plan


def test_example():
    config = load_config(Path(__file__).parents[1] / "configs/ursa.yaml")
    assert len(config.run.days()) == 151
    assert config.domains["conus"].shape == (2400, 6000)
    assert plan(config)["static"]


def test_period_leap_and_year_boundary():
    assert len(Period(start="2024-02-01", end="2024-03-01").days()) == 29
    assert len(Period(start="2026-02-01", end="2026-03-01").days()) == 28
    assert len(Period(start="2026-12-31", end="2027-01-02").days()) == 2


def test_roundtrip_and_templates(config, tmp_path):
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded == config
    assert render("{date}_{next_month}_{lead:03d}", config, date(2026, 12, 1), lead=3) == (
        "20261201_20270101_003"
    )
    with pytest.raises(ValueError, match="placeholder"):
        render("{unsupported}", config, date(2026, 1, 1))


@pytest.mark.parametrize(
    "update",
    [
        {"unexpected": 1},
        {"domains": {"bad": {"south": 0, "north": 2, "west": 0, "east": 2, "resolution": 0.3}}},
        {"distance_reference": [0, 0, -1, 2]},
    ],
)
def test_invalid_configuration(config, update):
    values = config.model_dump()
    values.update(update)
    with pytest.raises(ValidationError):
        Config.model_validate(values)


def test_reject_feature_and_training_leakage(config):
    values = config.model_dump()
    values["models"]["lstm"]["features"] = ["missing"]
    with pytest.raises(ValidationError, match="Feature"):
        Config.model_validate(values)
    values = config.model_dump()
    values["models"]["xgb_training"]["start"] = date(2026, 1, 8)
    with pytest.raises(ValidationError, match="follow LSTM"):
        Config.model_validate(values)


def test_tile_coverage_and_halos(config):
    grid = config.domains["tiny"]
    cover = np.zeros(grid.shape)
    for tile in tiles(grid):
        ys, xs = tile.slices(grid)
        cover[ys, xs] += 1
        target = target_grid(grid, tile)
        assert target.sizes["lat"] >= 2
        assert target.sizes["lon"] >= 2
    np.testing.assert_array_equal(cover, 1)
    lat, lon = coordinates(grid)
    np.testing.assert_allclose(lat, [2, 1.5, 1, 0.5])
    np.testing.assert_allclose(lon, [0, 0.5, 1, 1.5])
    valid, y, x = owners(grid, np.array([1.25, 0.5, 0]), np.array([0.75, 1.5, 0]))
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_array_equal(y[:2], [1, 3])
    np.testing.assert_array_equal(x[:2], [1, 3])


def test_degenerate_grid_and_distances():
    with pytest.raises(ValidationError):
        Grid(south=0, north=1, west=0, east=1, resolution=1)
    result = distances(np.array([49]), np.array([-125]), (-125, 25, -65, 49))
    assert result[0][0] == 0
    assert result[4][0] > 0


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("xgboost", "parameters", {"n_jobs": 100}),
        ("xgboost", "search", {"max_depth": []}),
        ("xgboost", "folds", 1),
        ("xgboost", "scoring", "not_a_scorer"),
        ("lstm", "units", []),
        ("lstm", "features", ["hour_utc", "hour_utc"]),
    ],
)
def test_invalid_model_settings_fail_before_submission(config, section, field, value):
    settings = config.model_dump()
    settings["models"][section][field] = value
    with pytest.raises(ValidationError):
        Config.model_validate(settings)


def test_monthly_template_cannot_silently_freeze_january(config):
    settings = config.model_dump()
    settings["variables"]["elevation"]["source"] = "/static/{month}.nc"
    with pytest.raises(ValidationError, match="static_period: monthly"):
        Config.model_validate(settings)


def test_scoring_default_and_roundtrip(config, tmp_path):
    settings = config.model_dump()
    del settings["models"]["xgboost"]["scoring"]
    assert Config.model_validate(settings).models.xgboost.scoring == "r2"
    config.models.xgboost.scoring = "neg_mean_absolute_error"
    path = tmp_path / "scoring.yaml"
    save_config(config, path)
    assert load_config(path).models.xgboost.scoring == "neg_mean_absolute_error"
