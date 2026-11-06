import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from dafcom.evaluate import metrics
from dafcom.models import chronological_folds, concentration, lstm_predict, window_starts


def test_windows_reject_internal_gaps_and_station_crossings():
    frame = pd.DataFrame(
        {
            "site_index": [1, 1, 1, 1, 2, 2, 2],
            "time_utc": pd.to_datetime(
                [
                    "2026-01-01T00",
                    "2026-01-01T01",
                    "2026-01-01T03",
                    "2026-01-01T04",
                    "2026-01-01T00",
                    "2026-01-01T01",
                    "2026-01-01T02",
                ]
            ),
        }
    )
    np.testing.assert_array_equal(window_starts(frame, 3), [4])
    np.testing.assert_array_equal(window_starts(frame, 2), [0, 2, 4, 5])


def test_scaling_and_output_shape(config):
    spec = config.models.lstm
    x = RobustScaler().fit(np.array([[0, 0, 0, 0], [2, 2, 2, 2]]))
    y = RobustScaler().fit(np.array([[2], [4]]))

    class Model:
        def predict(self, data, **kwargs):
            np.testing.assert_allclose(data, 0)
            return np.ones((len(data), spec.forecast_hours))

    data = np.ones((2, spec.lookback_hours, len(spec.features)))
    result = lstm_predict(Model(), {"x": x, "y": y}, data, spec)
    np.testing.assert_allclose(result, 4)
    np.testing.assert_allclose(concentration(result, spec), np.exp(4))
    with pytest.raises(ValueError, match="input shape"):
        lstm_predict(Model(), {"x": x, "y": y}, data[:, :1], spec)


def test_same_time_stations_never_cross_cv_split():
    times = np.repeat(pd.date_range("2026-01-01", periods=12, freq="h").values, 3)
    for train, test in chronological_folds(times, 3):
        assert set(times[train]).isdisjoint(set(times[test]))
        assert times[train].max() < times[test].min()


def test_metrics_exact():
    result = metrics([1, 2, 3], [2, 3, 4])
    assert result["mean_bias"] == 1
    assert result["mse"] == 1
    assert result["rmse"] == 1
    assert result["correlation"] == 1
    assert metrics([1], [2])["correlation"] is None
    with pytest.raises(ValueError):
        metrics([1], [np.nan])
    with pytest.raises(ValueError, match="paired"):
        metrics([1, 2], [1])


@pytest.mark.parametrize(
    "scoring", ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
)
def test_real_xgboost_chronological_search(config, monkeypatch, scoring):
    pytest.importorskip("xgboost")
    import json

    from dafcom import models
    from dafcom.models import artifact, load_xgb, train_xgb
    from dafcom.storage import product, write_csv

    config.models.xgboost.search = {"max_depth": [1, 2]}
    config.models.xgboost.search_iterations = 2
    config.models.xgboost.scoring = scoring
    original_search = models.RandomizedSearchCV
    selected_scorers = []

    def checked_search(*args, **kwargs):
        selected_scorers.append(kwargs["scoring"])
        return original_search(*args, **kwargs)

    monkeypatch.setattr(models, "RandomizedSearchCV", checked_search)
    for day in config.models.xgb_training.days():
        rows = []
        for time in pd.date_range(day, periods=24, freq="h"):
            for site in ["001", "1"]:
                rows.append(
                    {
                        "site_index": site,
                        "time_utc": time,
                        "lat": 1,
                        "lon": 1,
                        "v13_ufs_pm25": 10 + time.hour,
                        "v11_elevation": 1,
                        "obs_pm25": 12 + time.hour,
                    }
                )
        frame = pd.DataFrame(rows)
        write_csv(frame, product(config, "features", domain="tiny", day=day, suffix="csv"))
        stacked = frame[["site_index", "time_utc"]].copy()
        stacked["lstm_predict_log_pm25"] = np.log(frame.obs_pm25)
        stacked["lstm_pm25"] = frame.obs_pm25
        write_csv(stacked, product(config, "site_lstm", domain="tiny", day=day, suffix="csv"))
    train_xgb(config, "tiny")
    assert selected_scorers == [scoring]
    metadata_path = artifact(config, "tiny", "xgb_metadata", "json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["xgboost"]["scoring"] == scoring
    results = pd.read_csv(artifact(config, "tiny", "xgb_search", "csv"))
    assert len(results) == 2
    if scoring.startswith("neg_"):
        assert (results.mean_test_score <= 0).all()
    model = load_xgb(config, "tiny")
    assert model.n_jobs == 1
    assert np.isfinite(
        model.predict(
            stacked.assign(v13_ufs_pm25=10, v11_elevation=1)[config.models.xgboost.features]
        )
    ).all()
    if scoring == "r2":
        del metadata["xgboost"]["scoring"]
        metadata_path.write_text(json.dumps(metadata))
        load_xgb(config, "tiny")
    config.models.xgboost.scoring = "neg_mean_squared_error"
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_xgb(config, "tiny")


def test_full_forecast_pipeline(synthetic_data, tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("xgboost")
    import json

    import xarray as xr

    from dafcom.config import load_config
    from dafcom.grid import tiles
    from dafcom.storage import product
    from dafcom.workflow import local_run, prepare

    directory = prepare(synthetic_data, tmp_path / "complete")
    local_run(directory)
    config = load_config(directory / "config.yaml")
    day = config.models.prediction.start
    predictions = pd.read_csv(product(config, "site_xgb", domain="tiny", day=day, suffix="csv"))
    assert len(predictions) == 48
    assert np.isfinite(predictions.select_dtypes("number")).all().all()
    cover = np.zeros(config.domains["tiny"].shape)
    for tile in tiles(config.domains["tiny"]):
        with xr.open_dataset(
            product(config, "grid_predict", domain="tiny", day=day, tile=tile.name)
        ) as data:
            assert data.lstm_pm25.shape == (2, 2, 2)
            assert data.xgb_predictions_pm25.shape == (2, 2)
            assert np.isfinite(data.lstm_pm25).all()
            cover[tile.slices(config.domains["tiny"])] += 1
    np.testing.assert_array_equal(cover, 1)
    result = json.loads(product(config, "evaluation", domain="tiny", suffix="json").read_text())
    assert result["xgb_predictions_pm25"]["count"] == 48
    with xr.open_dataset(product(config, "forecast", domain="tiny", day=day)) as mosaic:
        assert mosaic.lstm_pm25.shape == (2, 4, 4)
        assert np.isfinite(mosaic.lstm_pm25).all()
        for tile in tiles(config.domains["tiny"]):
            with xr.open_dataset(
                product(config, "grid_predict", domain="tiny", day=day, tile=tile.name)
            ) as data:
                np.testing.assert_array_equal(
                    mosaic.lstm_pm25.values[(slice(None), *tile.slices(config.domains["tiny"]))],
                    data.lstm_pm25,
                )
    forecast = pd.read_csv(
        product(config, "site_lstm_forecast", domain="tiny", day=day, suffix="csv")
    )
    assert len(forecast) == 96
    assert set(forecast.lead_hour) == {0, 1}

    config.model_source = config.output_root / "models"
    config.models.site_predictions = False
    config.run.start = day - pd.Timedelta(days=1)
    config.run.end = config.models.prediction.end
    from dafcom.config import Config

    config = Config.model_validate(config.model_dump())
    operational = prepare(config, tmp_path / "forecast-only")
    local_run(operational)
    operational_config = load_config(operational / "config.yaml")
    assert not (operational_config.output_root / "observations").exists()
    with (
        xr.open_dataset(
            product(operational_config, "forecast", domain="tiny", day=day)
        ) as forecast_grid,
        xr.open_dataset(product(config, "forecast", domain="tiny", day=day)) as previous_grid,
    ):
        np.testing.assert_array_equal(forecast_grid.lstm_pm25, previous_grid.lstm_pm25)
        np.testing.assert_array_equal(
            forecast_grid.xgb_predictions_pm25, previous_grid.xgb_predictions_pm25
        )
    from dafcom.models import load_lstm, load_xgb

    incompatible = operational_config.model_copy(deep=True)
    incompatible.variables["pm25"].fields = ["different_field"]
    with pytest.raises(ValueError, match="scientific input"):
        load_lstm(incompatible, "tiny")
    with pytest.raises(ValueError, match="scientific input"):
        load_xgb(incompatible, "tiny")
