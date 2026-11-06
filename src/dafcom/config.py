"""Validated scientific and execution settings."""

from __future__ import annotations

import math
import string
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.metrics import get_scorer

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Period(Settings):
    start: date
    end: date

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Periods are half-open and require end > start")
        return self

    def days(self):
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days)]


class Grid(Settings):
    south: float = Field(ge=-90, le=90)
    north: float = Field(ge=-90, le=90)
    west: float = Field(ge=-180, le=180)
    east: float = Field(ge=-180, le=180)
    resolution: float = Field(gt=0)
    tile_shape: tuple[PositiveInt, PositiveInt] = (256, 256)

    @model_validator(mode="after")
    def rectangular(self):
        if self.north <= self.south or self.east <= self.west:
            raise ValueError("Require north > south and east > west; dateline crossing unsupported")
        for span in (self.north - self.south, self.east - self.west):
            cells = span / self.resolution
            if cells < 2 or not math.isclose(cells, round(cells), abs_tol=1e-7):
                raise ValueError("Grid spans must be integer multiples of resolution, >= 2 cells")
        return self

    @property
    def shape(self):
        return (
            round((self.north - self.south) / self.resolution),
            round((self.east - self.west) / self.resolution),
        )


class Variable(Settings):
    source: str
    fields: list[str] = Field(min_length=1)
    kind: Literal["dynamic", "static"] = "dynamic"
    static_period: Literal["fixed", "monthly"] = "fixed"
    feature: Name | None = None
    units: str = ""
    lat: str = "lat"
    lon: str = "lon"
    time: str = "time"
    indexers: dict[str, int] = Field(default_factory=dict)
    time_slice: tuple[int, int] | None = None
    file_per_hour: bool = False
    leads: list[int] = Field(default_factory=list)
    cycle_day_offsets: list[int] = Field(default_factory=lambda: [0])
    time_alignment: Literal["source", "legacy_index"] = "source"
    legacy_hour_offset: int = 0
    regrid_method: Literal["bilinear", "nearest_s2d"] = "bilinear"

    @model_validator(mode="after")
    def source_contract(self):
        if self.time_slice is not None and not (0 <= self.time_slice[0] < self.time_slice[1]):
            raise ValueError("time_slice requires 0 <= start < end")
        if self.file_per_hour and not self.leads:
            raise ValueError("file_per_hour requires explicit leads")
        if len(set(self.leads)) != len(self.leads):
            raise ValueError("Duplicate lead hours")
        if not self.cycle_day_offsets or len(set(self.cycle_day_offsets)) != len(
            self.cycle_day_offsets
        ):
            raise ValueError("cycle_day_offsets must be nonempty and unique")
        if self.time_alignment == "legacy_index" and self.cycle_day_offsets != [0]:
            raise ValueError("legacy_index requires cycle_day_offsets: [0]")
        if self.kind == "static" and (self.file_per_hour or self.leads):
            raise ValueError("Static sources cannot specify lead hours")
        if self.kind == "dynamic" and self.static_period != "fixed":
            raise ValueError("static_period applies only to static fields")
        if self.kind == "static" and self.static_period == "fixed":
            fields = {field for _, field, _, _ in string.Formatter().parse(self.source)}
            if fields & {"date", "year", "month", "day", "month_start", "next_month"}:
                raise ValueError("Time-varying static paths require static_period: monthly")
        return self


class Observations(Settings):
    source: str
    format: Literal["netcdf", "csv", "xlsx"] = "netcdf"
    variable: str = "PM2.5"
    time: str = "time"
    lat: str = "latitude"
    lon: str = "longitude"
    site: str | None = None
    indexers: dict[str, int] = Field(default_factory=dict)
    missing_values: list[float] = Field(default_factory=lambda: [-1])
    minimum: float = 0
    round_digits: int | None = Field(default=4, ge=0, le=12)


class LSTM(Settings):
    features: list[str] = Field(min_length=1)
    lookback_hours: PositiveInt = 96
    forecast_hours: PositiveInt = 24
    units: list[PositiveInt] = Field(default_factory=lambda: [84, 84], min_length=1)
    dense_units: PositiveInt = 28
    dropout: float = Field(default=0.2, ge=0, lt=1)
    epochs: PositiveInt = 10
    batch_size: PositiveInt = 48
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    target_transform: Literal["log", "identity"] = "log"
    seed: int = 42


class XGBoost(Settings):
    features: list[str] = Field(min_length=1)
    parameters: dict[str, float | int | str] = Field(
        default_factory=lambda: {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.15}
    )
    search: dict[str, list[float | int | str]] = Field(default_factory=dict)
    search_iterations: PositiveInt = 100
    scoring: str = "r2"
    folds: int = Field(default=10, strict=True, ge=2)
    seed: int = 42

    @field_validator("scoring")
    @classmethod
    def valid_scoring(cls, value: str) -> str:
        get_scorer(value)
        return value

    @model_validator(mode="after")
    def parameter_contract(self):
        forbidden = {"n_jobs", "nthread", "random_state"}
        if forbidden & (self.parameters.keys() | self.search.keys()):
            raise ValueError(
                "Use scheduler cpus and xgboost.seed instead of thread/seed parameters"
            )
        if any(not values for values in self.search.values()):
            raise ValueError("XGBoost search values cannot be empty")
        return self


class Models(Settings):
    lstm_training: Period
    xgb_training: Period
    prediction: Period
    lstm: LSTM
    xgboost: XGBoost
    model_variable: Name = "pm25"
    prediction_batch_cells: PositiveInt = 256
    grid_predictions: bool = True
    assemble_grid: bool = True
    site_predictions: bool = True


class Resources(Settings):
    cpus: PositiveInt = 4
    memory: str = Field(default="16G", pattern=r"^[1-9][0-9]*[KMGTP]?$")
    walltime: str = Field(default="02:00:00", pattern=r"^([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$")


class Slurm(Settings):
    account: Name = "arl"
    partition: Name = "u1-compute"
    qos: Name = "batch"
    max_array_size: PositiveInt = 1000
    max_concurrent: PositiveInt = 16
    max_dependency_jobs: int = Field(default=100, strict=True, ge=2)
    python: str = "python"
    setup: list[str] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)
    stage_resources: dict[str, Resources] = Field(default_factory=dict)

    @model_validator(mode="after")
    def lines(self):
        if not self.python or any(c in self.python for c in "\r\n"):
            raise ValueError("python must be a single executable path")
        if any("\n" in line or "\r" in line for line in self.setup):
            raise ValueError("Each setup entry must be one trusted shell command")
        return self


class Config(Settings):
    schema_version: Literal[1] = 1
    run: Period
    output_root: Path
    model_source: Path | None = None
    cycle_hour: int = Field(default=12, ge=0, le=23)
    domains: dict[Name, Grid] = Field(min_length=1)
    variables: dict[Name, Variable] = Field(min_length=1)
    observations: Observations
    models: Models
    slurm: Slurm = Field(default_factory=Slurm)
    distance_reference: tuple[float, float, float, float] = (-125, 25, -65, 49)
    baseline_variable: Name | None = None

    @model_validator(mode="after")
    def workflow_contract(self):
        periods = (
            (self.models.prediction,)
            if self.model_source is not None
            else (self.models.lstm_training, self.models.xgb_training, self.models.prediction)
        )
        if any(p.start < self.run.start or p.end > self.run.end for p in periods):
            raise ValueError("All model periods must lie within run")
        if self.models.lstm_training.end > self.models.xgb_training.start:
            raise ValueError("XGBoost stacking training must follow LSTM training")
        if self.models.xgb_training.end > self.models.prediction.start:
            raise ValueError("Prediction must follow both training periods")
        history = timedelta(hours=self.models.lstm.lookback_hours)
        first_inference = (
            self.models.prediction.start
            if self.model_source is not None
            else self.models.xgb_training.start
        )
        if datetime.combine(self.run.start, datetime.min.time()) + history > datetime.combine(
            first_inference, datetime.min.time()
        ):
            raise ValueError("run must include lookback history before inference")
        available = {v.feature for v in self.variables.values() if v.feature}
        derived = {
            "hour_utc",
            "day_of_year",
            "v20_lon",
            "v21_lat",
            "v22_d1",
            "v23_d2",
            "v24_d3",
            "v24_d4",
            "v25_d5",
        }
        features = available | derived
        reserved = derived | {
            "site_index",
            "time_utc",
            "lat",
            "lon",
            "obs_pm25",
            "log_pm25",
            "lstm_predict_log_pm25",
            "bias_correction_pm25",
        }
        if available & reserved:
            raise ValueError("Variable feature names must not override derived/reserved columns")
        if not self.models.site_predictions and not self.models.grid_predictions:
            raise ValueError("Enable site_predictions or grid_predictions")
        if self.observations.minimum < 0:
            raise ValueError("PM2.5 observation minimum must be nonnegative")
        for names, allowed in (
            (self.models.lstm.features, features),
            (self.models.xgboost.features, features | {"lstm_predict_log_pm25"}),
        ):
            if len(names) != len(set(names)) or not set(names) <= allowed:
                raise ValueError("Feature lists must be unique and reference configured features")
        if "lstm_predict_log_pm25" not in self.models.xgboost.features:
            raise ValueError("XGBoost features must include lstm_predict_log_pm25")
        if len(available) != sum(v.feature is not None for v in self.variables.values()):
            raise ValueError("Variable feature names must be unique")
        for name in (self.models.model_variable, self.baseline_variable):
            if name is not None and name not in self.variables:
                raise ValueError(f"Unknown model/baseline variable: {name}")
        if self.variables[self.models.model_variable].feature is None:
            raise ValueError("model_variable requires a feature name")
        if self.variables[self.models.model_variable].kind != "dynamic":
            raise ValueError("model_variable must be a dynamic field")
        if self.baseline_variable and self.variables[self.baseline_variable].kind != "dynamic":
            raise ValueError("baseline_variable must be a dynamic field")
        west, south, east, north = self.distance_reference
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("Invalid distance_reference bounds")
        render(self.observations.source, self, self.run.start)
        for template in (v.source for v in self.variables.values()):
            render(template, self, self.run.start, variable="test", lead=0)
        return self


def render(template: str, config: Config, day: date, **extra):
    next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    context = {
        "date": day.strftime("%Y%m%d"),
        "year": day.strftime("%Y"),
        "month": day.strftime("%m"),
        "day": day.strftime("%d"),
        "month_start": day.replace(day=1).strftime("%Y%m%d"),
        "next_month": next_month.strftime("%Y%m%d"),
        "cycle": config.cycle_hour,
        **extra,
    }
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None and field not in context:
            raise ValueError(f"Unsupported filename placeholder: {field}")
    return template.format_map(context)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = Config.model_validate(yaml.safe_load(stream))
    config.output_root = (path.parent / config.output_root).resolve()
    if config.model_source is not None:
        config.model_source = (path.parent / config.model_source).resolve()
    for variable in config.variables.values():
        variable.source = str(path.parent / variable.source)
    config.observations.source = str(path.parent / config.observations.source)
    return config


def save_config(config: Config, path: Path):
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
