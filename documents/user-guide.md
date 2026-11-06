# DAFCOM user guide: configuration, training, evaluation, and Ursa forecasting

This guide describes the executable package on the `ursa_2026` branch. It does
not use the legacy date-specific Python files or their submission shell loops.
The legacy forecast and reanalysis trees are no longer included in this branch;
consult the
[pre-removal source snapshot](https://github.com/drnimbusrain/DAFCOM/tree/bddff9f9593759bdbe5047e428aa49627d3ce8c0/src)
when historical reference scripts are needed. Reanalysis research workflows are
not implemented by the optimized forecast package.
For the scientific changes relative to those scripts, see the
[migration guide](modernization.md). The reference configuration is
[`configs/ursa.yaml`](../configs/ursa.yaml).

## Contents

1. [Choose a workflow mode](#1-choose-a-workflow-mode)
2. [Install on Ursa](#2-install-on-ursa)
3. [Create and validate your configuration](#3-create-and-validate-your-configuration)
4. [Configure dates and paths](#4-configure-dates-and-paths)
5. [Configure domains and parallel tiles](#5-configure-domains-and-parallel-tiles)
6. [Configure UFS and static predictors](#6-configure-ufs-and-static-predictors)
7. [Configure observations](#7-configure-observations)
8. [Configure LSTM and XGBoost](#8-configure-lstm-and-xgboost)
9. [Train and evaluate on Ursa](#9-train-and-evaluate-on-ursa)
10. [Evaluate previously trained models](#10-evaluate-previously-trained-models)
11. [Run observation-free gridded forecasts](#11-run-observation-free-gridded-forecasts)
12. [Configure Slurm resources and monitor jobs](#12-configure-slurm-resources-and-monitor-jobs)
13. [Understand outputs](#13-understand-outputs)
14. [Troubleshooting and recovery](#14-troubleshooting-and-recovery)
15. [Acceptance checklist](#15-acceptance-checklist)

## 1. Choose a workflow mode

There are no CLI subcommands named `train`, `evaluate`, or `forecast`. The
configuration and optional `--stages` determine which workflow is prepared.

| Purpose | `model_source` | Main settings/selection | Observations needed? |
|---|---|---|---|
| Prepare station features | Usually `null` | `--stages features` | Yes |
| Train models and evaluate withheld dates | `null` | Default stages; `site_predictions: true` | Yes |
| Train LSTM only | `null` | `--stages train_lstm` | Yes |
| Train LSTM and XGBoost without final evaluation | `null` | `--stages train_xgb` | Yes |
| Evaluate existing models on new dates | Previous `products/models/` | `--stages evaluate` | Yes |
| Produce gridded forecasts with existing models | Previous `products/models/` | `site_predictions: false`, `grid_predictions: true` | No |

Important current capabilities and limits:

- The modeled target is PM2.5. Other chemical variables can be predictors.
- Models are trained **per configured domain**, not independently per tile.
- Grid LSTM forecasts are issued once per prediction day at **00 UTC**.
  `cycle_hour` selects UFS source filenames; it does not change this issuance.
- The default LSTM uses 96 hours of predictors to predict the next 24 hours.
- Gridded XGBoost refinement is only for the **first valid forecast hour**.
  It is not a 24-hour XGBoost product.
- Ingestion currently requires a complete 24-hour valid-time input day for
  every day in `run`, including each prediction day. This is a batch/archive
  interface, not a real-time data-availability or cycle-launch service.
  Ensure selected UFS files are actually available at the proposed launch time
  and do not introduce future information into a claimed real-time evaluation.
- There is no automatic downloading, HPSS retrieval, model registry, scheduled
  daily launch, or automatic failed-job restart.

## 2. Install on Ursa

### 2.1 Prepare a shared environment

Use the site's supported Python/ESMF environment. Python 3.12 is the tested ML
environment. Core Python support is 3.10+, but TensorFlow and ESMF compatibility
must be checked separately.

Put the environment, installed package, source data, and run directory on
filesystems accessible from compute nodes. The following paths are placeholders;
replace them with your assigned locations:

```bash
cd /path/to/DAFCOM

# After loading site-approved Python/ESMF modules:
python --version
python -m venv /shared/path/envs/dafcom
source /shared/path/envs/dafcom/bin/activate

python -m pip install -e '.[ml,regrid]'
```

For curvilinear UFS grids, ESMF/esmpy must already be available to this Python
environment. Installing xESMF with pip does **not** install esmpy. Use the
Ursa-supported module or environment solution; module names are deliberately
not prescribed here. A virtual environment must be able to import that site's
esmpy installation.

An editable installation is useful during development, but compute jobs then
depend on the checkout. Do not edit that checkout while jobs are running.
For a fixed operational candidate, build a wheel and install it into a dedicated
environment:

```bash
python -m pip wheel --no-deps . --wheel-dir /shared/path/wheels
python -m pip install \
  '/shared/path/wheels/dafcom-0.1.0-py3-none-any.whl[ml,regrid]'
```

Use the wheel filename produced by your build. Configuration/task checksums do
not freeze the installed Python code or external source data.

### 2.2 Verify dependencies

```bash
python -c 'import dafcom; print(dafcom.__version__, dafcom.__file__)'
python -c 'import tensorflow, xgboost, xarray, netCDF4'
python -c 'import xesmf'
dafcom --help
```

The xESMF check is required for curvilinear sources, not purely rectilinear
synthetic runs. XLSX observation input additionally requires:

```bash
python -m pip install -e '.[excel]'
```

For development tests:

```bash
python -m pip install -e '.[test,ml]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
python -m pytest -q
```

Do not run full-domain training or data processing on a login node. Use Slurm
or an appropriately sized interactive allocation.

## 3. Create and validate your configuration

The configuration filename is arbitrary; `config.yaml` is not loaded implicitly.
Always pass its path. Keep a working copy next to the reference so relative
paths retain the same meaning:

```bash
cp configs/ursa.yaml configs/config.yaml
# Edit configs/config.yaml with your paths, dates, domains, and resources.
dafcom validate configs/config.yaml
dafcom plan configs/config.yaml
```

The guide's YAML snippets are **edits to the complete reference configuration**,
not independent complete files. Do not replace the entire file with a snippet.

| Command | Effect |
|---|---|
| `dafcom validate CONFIG` | Validate settings and filename-template syntax; no processing/submission. |
| `dafcom plan CONFIG` | Show task counts, including prerequisites; no processing/submission. |
| `dafcom prepare CONFIG --run-dir DIRECTORY` | Create snapshots and batch scripts in a new directory; no submission. |
| `dafcom submit --run-dir DIRECTORY` | Submit that prepared workflow to Slurm. |
| `dafcom run CONFIG --workers N` | Prepare a new workflow and execute locally using processes. |
| `dafcom worker ...` | Internal array-task entry point; normally invoked by generated scripts. |

Global logging options go before the subcommand:

```bash
dafcom --log-level DEBUG plan configs/config.yaml
```

`validate` does not open source files, test Slurm accounts, verify available disk
space, or prove that pretrained artifacts match. A valid configuration is not
a data-readiness check. Also run `plan`, which checks stage-resource names.

`--stages` selects terminal stages **plus all their prerequisites**; it does not
skip prerequisites or reuse arbitrary products from another run. Every `run`
or `prepare` creates isolated products in a new workflow directory.

## 4. Configure dates and paths

### 4.1 Top-level settings

```yaml
schema_version: 1
run: {start: 2026-01-01, end: 2026-06-01}
output_root: /shared/scratch/dafcom
model_source: null
cycle_hour: 12
distance_reference: [-125, 25, -65, 49]
baseline_variable: null
```

| Setting | Meaning |
|---|---|
| `schema_version` | Must be `1`. Unknown configuration keys are rejected. |
| `run.start`, `run.end` | Input processing interval, in whole UTC days, `[start, end)`. |
| `output_root` | Parent for automatically named workflows when `--run-dir` is omitted. |
| `model_source` | `null` for training; otherwise the parent of domain-specific trained artifacts. |
| `cycle_hour` | Integer 0-23, substituted for `{cycle}` in source templates. |
| `distance_reference` | `[west, south, east, north]` bounds for optional distance features. |
| `baseline_variable` | Name of a configured dynamic bias-correction field, or `null`. |

With `--run-dir`, products go under `DIRECTORY/products/`, not directly into
`output_root`. Without it, a unique workflow is created under
`output_root/workflows/`.

All intervals exclude the end date: `2026-05-01` to `2026-06-01` includes every
day in May, not June 1. Calendar rollover and leap days are generated, not listed
manually.

For training, all three model periods must lie within `run`, with:

```text
LSTM training end <= XGBoost training start
XGBoost training end <= prediction start
```

Include at least `lookback_hours` of input history before the first XGBoost
training day. For model reuse, include that history before the first prediction
day; original training periods may then lie outside `run`.

### 4.2 Filename templates

Relative paths resolve against the original YAML file's directory. Absolute
paths are preferable for shared Ursa datasets. Shell expansion such as `$USER`
or `~` is not performed by the configuration loader.

| Placeholder | Example for May 1, 2026 |
|---|---|
| `{date}` | `20260501` |
| `{year}`, `{month}`, `{day}` | `2026`, `05`, `01` |
| `{month_start}` | `20260501` |
| `{next_month}` | `20260601` |
| `{cycle:02d}` | `12` for `cycle_hour: 12` |
| `{variable}` | Configured variable key; predictor sources only. |
| `{lead:03d}` | `001`, `002`, etc.; predictor sources only. |

`{next_month}` handles December-to-January correctly. Observation templates do
not accept `{variable}` or `{lead}`.

YAML anchors such as `&phy` and `<<: *phy` in the reference reduce repetition.
A change to the anchored mapping affects inheriting entries unless overridden.
Quote `"no"` and similar YAML 1.1 boolean-like strings.

## 5. Configure domains and parallel tiles

```yaml
domains:
  east_conus:
    south: 25
    north: 49
    west: -95
    east: -65
    resolution: 0.01
    tile_shape: [256, 256]
  west_conus:
    south: 25
    north: 49
    west: -125
    east: -95
    resolution: 0.01
    tile_shape: [256, 256]
```

Domain and variable keys must start with a letter and contain only letters,
digits, underscores, or hyphens.

- `resolution` is degrees, not kilometers.
- Each span divided by resolution must be an integer with at least two cells.
- Latitude runs north-to-south; longitude west-to-east. South/east endpoints
  are excluded: east CONUS above has shape `2400 x 3000`.
- `tile_shape` is `[latitude_cells, longitude_cells]`. Edge tiles may be smaller.
- Tiles have nonoverlapping output ownership. Regridded tiles additionally have
  a one-cell halo for station interpolation across tile boundaries.
- Dateline-crossing domains and moving source grids are unsupported.

Smaller tiles reduce task memory but increase task counts and filesystem load.
`prediction_batch_cells` separately bounds the number of grid cells sent through
prediction at once.

Adding a domain means training another domain-specific model in training mode.
Pretrained mode requires the same domain name and bounds/resolution as its
artifacts. A model trained for `conus` cannot simply be loaded as `east_conus`.
Tile size can change without changing the trained grid contract.

For a canary, choose a small domain that contains several observation sites,
not merely one grid cell. Train a separate canary model; it does not replace a
CONUS model.

### 5.1 How a domain becomes independently executable tiles

A named domain defines the complete geographic target grid. A tile is a
rectangular processing unit within that grid, identified by integer start/stop
row and column indices. Tiles are not separate experiments or separately
trained models.

For the reference full CONUS domain:

| Quantity | Value |
|---|---|
| Grid dimensions | 2,400 latitude rows x 6,000 longitude columns |
| Total cells | 14,400,000 |
| Configured tile dimensions | 256 x 256 cells |
| Tile layout | 10 tile rows x 24 tile columns |
| Total tiles | 240 |

Tile counts round up in each direction; outer tiles are smaller where the
domain dimensions are not exact multiples of the tile dimensions. Every target
cell belongs to exactly one core tile, with no gaps or duplicate ownership.

Each tile task identifies its date, named domain, row/column bounds, and shared
configuration. Once prerequisite inputs and models are available, regridding
or prediction for that tile can execute without waiting for neighboring tiles.
Slurm distributes these tasks across available workers, subject to the
configured array concurrency limit. Each worker writes a separate tile product,
avoiding concurrent writes to the same NetCDF file.

All prediction tiles within a domain use the same domain-trained models,
scalers, and feature ordering. Tiling changes the execution strategy, not the
scientific model assigned to each cell.

### 5.2 Why regridded tiles include a halo

Station interpolation near a tile edge needs surrounding grid values that may
lie just outside the tile's core. Regridded tiles therefore include a one-cell
halo, clipped at domain boundaries. This small overlap supplies those neighboring
values without requiring workers to exchange data during station sampling.

Final gridded forecasts contain only core cells, not duplicated halo cells.
After tile prediction completes, assembly checks each tile's coordinates,
times, shapes, and finite values, then writes it into its assigned portion of
the domain NetCDF file.

### 5.3 How tiling and batching reduce memory

A full-domain predictor tensor containing every CONUS cell, 96 historical
hours, and 21 predictors would hold:

```text
14,400,000 cells x 96 hours x 21 predictors x 4 bytes = 116.1 GB
```

This assumes float32 values and decimal GB. It describes only that raw tensor,
not total process memory; scaling copies, intermediate arrays, model weights,
and predictions require additional space.

A full 256 x 256 tile contains 65,536 cells, about 220 times fewer cells than
the full domain. Prediction further subdivides each tile into cell batches:

| Predictor tensor scope | Cells | Memory for 96 hours x 21 float32 predictors |
|---|---:|---:|
| Full CONUS | 14,400,000 | Approximately 116.1 GB |
| One full tile | 65,536 | Approximately 528.5 MB |
| One prediction batch | 256 | Approximately 2.1 MB |

With `prediction_batch_cells: 256`, the implementation constructs histories for
256 cells at a time rather than allocating the full-tile or full-domain
predictor tensor. Tile forecast output arrays remain in memory, along with
models, scalers, numerical-library workspace, and I/O buffers. The 2.1 MB figure
is therefore **not** a job-memory recommendation.

Assembly also bounds memory: it reads one forecast tile at a time and writes
directly into the domain output, rather than holding the entire domain forecast
in RAM.

### 5.4 Processing benefits and practical trade-offs

- **Parallel execution:** independent tile tasks can run on multiple nodes.
- **Manageable allocations:** smaller target-grid working sets can fit
  lower-memory jobs and available compute resources more easily.
- **Separate controls:** `tile_shape` controls spatial task size,
  `prediction_batch_cells` controls predictor-batch size, and
  `slurm.max_concurrent` controls concurrent array tasks per stage.
- **Failure localization:** date/domain/tile outputs and logs identify the
  affected work unit. Automatic retry/resume is not currently implemented.
- **Consistent domain products:** verified core tiles assemble without seams
  caused by duplicate cell ownership or missing tile placement.

Tiling reduces per-task target-grid memory and enables parallelism; it does not
eliminate the total workload or guarantee a particular speedup. Smaller tiles
create more files, scheduler tasks, repeated reads, and scheduling overhead.
Each prediction worker loads its own models. Independent stages can overlap,
so a per-stage concurrency limit is not a workflow-wide memory or CPU cap.

Current regridding tasks can still load complete native source fields, even
though their target grids are tiled. Ingestion, station-feature processing, and
training have separate memory requirements and are not all controlled by
`prediction_batch_cells`. Tiling also does not reduce the final domain's cell
count or automatically reduce retained storage.

Choose tile and batch sizes using representative Ursa canaries, balancing
measured peak memory and walltime against scheduler and shared-filesystem
overhead.

## 6. Configure UFS and static predictors

Each key under `variables` names a package field, its output filenames, and
optionally its ML feature. All configured dynamic fields are ingested and
regridded, including fields with `feature: null`.

### 6.1 Dynamic source example

```yaml
variables:
  t2m:
    source: /archive/phy/{date}/aqm.{date}.t{cycle:02d}z.phy.f{lead:03d}.nc
    fields: [tmp2m]
    feature: v5_t2m
    units: K
    lat: lat
    lon: lon
    time: time
    file_per_hour: true
    leads: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    cycle_day_offsets: [-1, 0]
    time_alignment: source
    regrid_method: bilinear
```

| Variable setting | Default | Description |
|---|---|---|
| `source` | Required | Path/template; predictor file discovery supports glob patterns. |
| `fields` | Required | One field, or fields summed elementwise, with identical dimensions. |
| `kind` | `dynamic` | `dynamic` or `static`. |
| `feature` | `null` | Unique name referenced in ordered ML feature lists. |
| `units` | Empty | Output unit label; does not perform unit conversion. |
| `lat`, `lon`, `time` | `lat`, `lon`, `time` | Source coordinate names. |
| `indexers` | `{}` | Integer dimension selections, e.g. `{layer: 0}`. |
| `time_slice` | `null` | `[start_index, stop_index)` for a multi-time source. |
| `file_per_hour` | `false` | Require one file for each explicit lead when `true`. |
| `leads` | `[]` | Ordered unique lead-hour integers for per-hour files. |
| `cycle_day_offsets` | `[0]` | Dates relative to the UTC input day used to resolve source paths. |
| `time_alignment` | `source` | Decoded valid times, or explicit `legacy_index` compatibility mode. |
| `legacy_hour_offset` | `0` | Starting indexed hour in legacy mode. |
| `regrid_method` | `bilinear` | `bilinear` or `nearest_s2d`. |
| `static_period` | `fixed` | Static fields only: `fixed` or `monthly`. |

In source mode, the selected files must supply **exactly one record for each
UTC hour 00-23** after filtering to the input day. Duplicate, missing, or
non-datetime valid times fail.

For a 12Z forecast with leads 1-24, previous/current cycle dates can provide
complete UTC days. With `[-1, 0]`, the day before `run.start` must be available.
Inspect actual source valid times; do not assume sorted filenames imply a
particular valid hour. If overlapping cycles supply duplicate valid hours,
adjust lead/time selections rather than relying on an implicit preference.

### 6.2 Surface chemistry and summed fields

```yaml
pm25:
  source: /archive/chem/{date}/aqm.total.chem_sfc.nc
  fields: [PM25_TOT]
  feature: v13_ufs_pm25
  units: ug/m3
  time_slice: [0, 24]
  cycle_day_offsets: [-1, 0]
  time_alignment: source
  indexers: {layer: 0}  # Replace "layer" with the actual source dimension.
```

`time_slice` is positional selection, not a date range. Nonsingleton vertical
dimensions require explicit `indexers`; extra singleton surface dimensions can
be squeezed. For evaporation, `fields: [evbs_ave, evcw_ave]` sums the two fields.
There is no general formula interpreter.

### 6.3 Fixed and monthly static fields

```yaml
land_use:
  source: /static/LUC_0p01_regrid_data.nc
  fields: [luc]
  kind: static
  static_period: fixed
  feature: v10_luc
  regrid_method: nearest_s2d

e_bc:
  source: /static/emissions/2019/2019_{month}/E_bc_0p01_regrid_data.nc
  fields: [E_bc]
  kind: static
  static_period: monthly
  feature: v14_e_bc
```

Fixed maps are processed once per domain/tile. Monthly maps are processed once
per month/domain/tile. A temporal path placeholder in a fixed static field is
rejected to avoid silently using January's map all year.

Select non-singleton static time dimensions explicitly, for example
`indexers: {time: 0}`, if one representative map is scientifically appropriate.
Set units to the actual data units; verify that summed fields have compatible
units. Nearest-neighbor regridding is generally appropriate for categorical
land-use classes.

Rectilinear sources use xarray interpolation. Curvilinear sources use xESMF;
weights are reused across variables sharing the same source/target grid and
method within a task. Unmapped/nonfinite features are not silently filled.

## 7. Configure observations

### 7.1 AirNow NetCDF

```yaml
observations:
  source: /archive/AirNow/{year}/AirNow_{month_start}_{next_month}.nc
  format: netcdf
  variable: PM2.5
  time: time
  lat: latitude
  lon: longitude
  site: station_id
  indexers: {}
  missing_values: [-1]
  minimum: 0
  round_digits: 4
```

Use the actual station ID variable name. `site: null` uses positional indices;
this is safe only when source ordering remains stable across files.

Missing sentinels are masked **before** hourly averaging. Finite hourly values
strictly greater than `minimum` are retained, cropped to the day and domain,
and rounded if `round_digits` is not `null`. The default excludes zero/negative
PM2.5, which cannot be log-transformed. Very small positive values must not be
rounded to zero; increase precision, disable rounding, or choose an appropriate
quality threshold.

Observation keys are station/hour. Missing IDs and duplicate keys are errors.
Station coordinates must stay consistent across input tables. Prefer stable
string identifiers so leading zeros are meaningful and preserved.

### 7.2 CSV or XLSX

For an input table containing `station`, `valid`, `latitude`, `longitude`, and
`observed_pm25`, use:

```yaml
observations:
  source: /archive/observations_{date}.csv
  format: csv
  variable: observed_pm25
  time: valid
  lat: latitude
  lon: longitude
  site: station
  missing_values: [-1]
  minimum: 0
  round_digits: 4
```

Timestamps should explicitly represent UTC; offset-aware timestamps are converted
to UTC. Change `format` to `xlsx` and install the `excel` extra for Excel input.

Even observation-free gridded mode requires an `observations` configuration
section for schema validation, but it does not open those observation files.

## 8. Configure LSTM and XGBoost

### 8.1 Training and prediction periods

```yaml
models:
  lstm_training: {start: 2026-01-01, end: 2026-04-01}
  xgb_training: {start: 2026-04-01, end: 2026-05-01}
  prediction: {start: 2026-05-01, end: 2026-06-01}
  model_variable: pm25
  site_predictions: true
  grid_predictions: true
  assemble_grid: true
  prediction_batch_cells: 256
```

`model_variable` must reference a dynamic variable with a feature name; its
sampled value is the UFS PM2.5 baseline. `site_predictions` enables the default
site-prediction/evaluation graph. `grid_predictions` enables map generation;
`assemble_grid` adds streamed per-domain assembly. At least one prediction
flag must be true even when explicitly selecting only training stages.

The example reserves January-March for LSTM, April for XGBoost, and May for
withheld evaluation. No training interval may overlap a later fitting/prediction
interval. Gaps are allowed; include necessary history in `run`.

### 8.2 LSTM settings

Keep the complete ordered feature list from the reference, or deliberately
replace it with names defined under `variables` plus supported derived names.

| Setting | Default | Meaning |
|---|---|---|
| `features` | Required | Ordered predictor names; excludes the observation target. |
| `lookback_hours` | `96` | Contiguous predictor history per sample. |
| `forecast_hours` | `24` | Number of subsequent hourly targets/output values. |
| `units` | `[84, 84]` | Nonempty list of LSTM layer widths. |
| `dense_units` | `28` | Intermediate dense layer width. |
| `dropout` | `0.2` | Dropout fraction, 0 inclusive to 1 exclusive. |
| `epochs` | `10` | Training epochs. |
| `batch_size` | `48` | Sequence batch size for training/prediction. |
| `validation_fraction` | `0.2` | Final fraction of the training time interval held out. |
| `target_transform` | `log` | `log` or `identity` target transform. |
| `seed` | `42` | Training seed; not a promise of bitwise cross-platform reproducibility. |

For log targets, inputs and targets use separate fitted RobustScalers. Both site
and grid prediction apply X scaling and inverse Y scaling consistently.
Validation windows begin after the chronological cutoff; training windows end
before it. Overlapping histories across that boundary are excluded.
Scalers fit only pre-cutoff rows.

There must be enough continuous station data for both training and validation
windows of `lookback_hours + forecast_hours`. Missing internal hours break a
window; station sequences never cross between sites.

Available derived features include:

- `hour_utc`, `day_of_year`;
- `v20_lon`, `v21_lat`;
- `v22_d1`, `v23_d2`, `v24_d3`, `v24_d4`, `v25_d5`: distances to the
  northwest, northeast, southwest, southeast, and center reference points.

The historical distance names are intentionally preserved, including `v24_d4`
and `v25_d5`. Do not define variable features using reserved/derived names.

### 8.3 XGBoost settings

`models.xgboost.features` is another ordered feature list and must include
`lstm_predict_log_pm25`. This feature always denotes log concentration, including
when the LSTM target uses `identity` (which then requires positive predictions).

```yaml
xgboost:
  # Retain the full features list from the reference configuration.
  features: [v13_ufs_pm25, lstm_predict_log_pm25, hour_utc, day_of_year]
  parameters:
    n_estimators: 100
    max_depth: 3
    learning_rate: 0.15
  search: {}
  search_iterations: 100
  scoring: r2
  folds: 10
  seed: 42
```

The shortened feature list above is an example of a deliberate alternative
model, not equivalent to the reference model.

An empty `search` fits the configured estimator directly. Enable randomized
search with a nonempty parameter mapping:

```yaml
search:
  max_depth: [3, 5, 7]
  n_estimators: [100, 200]
  learning_rate: [0.05, 0.15]
search_iterations: 6
scoring: neg_root_mean_squared_error
folds: 5
```

Search uses chronological valid-time folds; all stations at a given hour remain
on the same side of each split. Ensure enough unique times for the requested
folds. `scoring` selects a scikit-learn scorer by name and defaults to `r2`.
Invalid scorer names fail configuration validation. Common regression choices:

| `scoring` | Selection criterion |
|---|---|
| `r2` | Maximize R-squared (default). |
| `neg_root_mean_squared_error` | Minimize RMSE. |
| `neg_mean_squared_error` | Minimize MSE. |
| `neg_mean_absolute_error` | Minimize MAE. |

Search always maximizes the score, so error metrics require their `neg_` names.
For example, a saved negative RMSE score of `-5` is better than `-8`; negate it
to report a positive RMSE. Use a regression scorer appropriate to continuous
PM2.5 targets, not a classification scorer.

`scoring` applies only when `search` is nonempty. It does not change the
estimator's fitting objective or the final external evaluation metrics.
The scorer is recorded in model metadata and checked on model reuse; retain
the trained value in forecast configurations. Artifacts created before this
setting existed are treated as using `r2`.

Search is serial across candidate fits; each estimator uses
the stage's allocated CPUs. Do not put `n_jobs`, `nthread`, or `random_state` in
parameters/search; use Slurm `cpus` and `seed`.

## 9. Train and evaluate on Ursa

### 9.1 Recommended combined run

Start from the full reference YAML, with your archive paths and domain choices.
Use `model_source: null`. Set the three model periods as in section 8 and
`run` to cover them plus any required history.

For station-only training/evaluation without maps:

```yaml
models:
  # Keep all other models settings.
  site_predictions: true
  grid_predictions: false
```

This avoids expensive grid-prediction/assembly tasks but still regrids input
tiles needed for station sampling. Enable both flags if maps are also required.

```bash
dafcom validate configs/config.yaml
dafcom plan configs/config.yaml

dafcom prepare configs/config.yaml \
  --run-dir /shared/scratch/dafcom/train-202601-may-eval

# Inspect snapshots, task counts, and Slurm resource requests before submission.
less /shared/scratch/dafcom/train-202601-may-eval/config.yaml
less /shared/scratch/dafcom/train-202601-may-eval/manifest.json
less /shared/scratch/dafcom/train-202601-may-eval/train_lstm/0.slurm

dafcom submit \
  --run-dir /shared/scratch/dafcom/train-202601-may-eval
```

Replace `/shared/scratch/...` with a real assigned Ursa path. The destination
must not already exist. Confirm queue/account/resource limits before submission.

Conceptually, this executes:

```text
ingest -> regrid ----+
static -------------+-> features -> train_lstm -> site_lstm -> train_xgb
observations -------+                               |              |
                                                    +----> site_xgb -> evaluate
train_xgb + regrid + static -> grid_predict -> assemble  (if enabled)
```

Dependencies are stage barriers, not day-by-day pipelining. A model stage waits
for its complete prerequisite stage, including all its array chunks.

### 9.2 Partial/training-only selections

```bash
# Inspect a station-feature pipeline without model training.
dafcom plan configs/config.yaml --stages features

# Train only the LSTM, including input and feature prerequisites.
dafcom prepare configs/config.yaml --stages train_lstm \
  --run-dir /shared/scratch/dafcom/lstm-only

# Train both models; omit final site_xgb/evaluate/grid_predict stages.
dafcom prepare configs/config.yaml --stages train_xgb \
  --run-dir /shared/scratch/dafcom/model-training
```

Submit each prepared directory explicitly when ready. Selecting `train_xgb`
also selects `site_lstm`, whose tasks currently cover **both** XGBoost training
and prediction periods. It is not a minimal "April-only" stacking task plan.
All selected feature/input stages likewise cover the complete `run` interval.

An LSTM-only artifact directory is insufficient for the current pretrained
combined forecast path, which also loads XGBoost. There is no CLI mode to reuse
an existing LSTM while training only a new XGBoost model.

### 9.3 Local canary

For a small configuration inside an adequately sized interactive allocation:

```bash
dafcom run configs/config.yaml --stages features --workers 2 \
  --run-dir /shared/scratch/dafcom/features-canary
```

`run` is not a Slurm submission command. It executes on the current host.
Process workers isolate NetCDF/HDF5 and ML libraries. Budget at least the number
of worker processes times the per-task CPU/memory requirements. There is no
automatic reuse of products created by a previous canary.

## 10. Evaluate previously trained models

Use this mode to avoid retraining when assessing a later withheld interval.

1. Copy the completed training run's snapshot, not an unrelated experiment's
   configuration.
2. Point `model_source` at the training run's `products/models/` directory.
3. Keep the original LSTM/XGBoost settings and training periods unchanged.
4. Change only input/prediction dates, compatible source paths, and execution
   settings as needed.
5. Retain observations for input history **and** the evaluation period.

```bash
cp /shared/scratch/dafcom/train-202601-may-eval/config.yaml \
  configs/evaluation.yaml
```

Edit these sections of that complete copy:

```yaml
model_source: /shared/scratch/dafcom/train-202601-may-eval/products/models
output_root: /shared/scratch/dafcom
run: {start: 2026-06-01, end: 2026-06-08}
models:
  # Keep original training periods, model features, and hyperparameters.
  prediction: {start: 2026-06-05, end: 2026-06-08}
  site_predictions: true
  grid_predictions: false
```

This includes four days of history before June 5 for the default 96-hour LSTM.
The evaluation period is June 5-7.

```bash
dafcom validate configs/evaluation.yaml
dafcom plan configs/evaluation.yaml --stages evaluate
dafcom prepare configs/evaluation.yaml --stages evaluate \
  --run-dir /shared/scratch/dafcom/evaluate-june05-07
dafcom submit --run-dir /shared/scratch/dafcom/evaluate-june05-07
```

With `model_source` set, neither training stage appears. This still performs
input processing, observations, feature sampling, and inference in the new run.
It is not a command for merely recalculating metrics on an arbitrary old CSV.

### 10.1 Evaluation interpretation

For each domain, evaluation compares observed PM2.5 with:

- `xgb_predictions_pm25`;
- `lstm_pm25` from the first predicted hour of each station issuance;
- `ufs_pm25`;
- optional `bias_correction_pm25`.

It reports paired count, correlation, MSE, RMSE, MAE, mean bias, model standard
deviation, and observation standard deviation. Mean bias is prediction minus
observation. Correlation is JSON `null` for singleton/constant series, with a
warning. A scatter diagnostic is also saved.

These are pooled matched station/hour statistics, not automatically
site-stratified or all-lead verification. Rows lacking contiguous LSTM histories
can be excluded from stacking with a warning. Examine counts and missingness,
not just RMSE. Empty/nonfinite evaluation inputs fail.

### 10.2 Compare an external bias-correction field

Before training, add a dynamic variable with its real source field/time contract:

```yaml
variables:
  bias_corrected_pm25:
    source: /archive/bias_correction/{date}.nc
    fields: [pm25_bc]
    units: ug/m3
    feature: null
    time_alignment: source
baseline_variable: bias_corrected_pm25
```

The package samples it on the same station/hour keys and includes it in
evaluation. Adding/removing a variable changes the scientific artifact contract,
even with `feature: null`; configure this consistently before training and
model reuse rather than adding it only to a pretrained evaluation configuration.

## 11. Run observation-free gridded forecasts

### 11.1 Create a forecast configuration

Copy the completed training snapshot:

```bash
cp /shared/scratch/dafcom/train-202601-may-eval/config.yaml \
  configs/forecast.yaml
```

Edit these parts of the complete copy:

```yaml
model_source: /shared/scratch/dafcom/train-202601-may-eval/products/models
output_root: /shared/scratch/dafcom
run: {start: 2026-06-01, end: 2026-06-06}
models:
  # Keep original training periods, LSTM settings, and XGBoost settings.
  prediction: {start: 2026-06-05, end: 2026-06-06}
  site_predictions: false
  grid_predictions: true
  assemble_grid: true
  prediction_batch_cells: 256
```

This creates one grid issuance at `2026-06-05 00:00 UTC`. With the default
lookback, the LSTM consumes June 1 00 UTC through June 4 23 UTC.
Its 24 outputs are valid June 5 00-23 UTC. XGBoost consumes the first LSTM output
plus June 5 00 UTC environmental predictors and refines only that first hour.

Despite this narrow inference window, the ingestion stage currently processes
**all** 24 hours for each `run` day, including June 5. Verify availability of
that complete input window; do not describe this as an arbitrary-cycle real-time
forecast launch.

### 11.2 Inspect and submit

```bash
dafcom validate configs/forecast.yaml
dafcom plan configs/forecast.yaml
dafcom prepare configs/forecast.yaml \
  --run-dir /shared/scratch/dafcom/forecast-20260605
dafcom submit --run-dir /shared/scratch/dafcom/forecast-20260605
```

The default forecast-only plan contains:

```text
ingest -> regrid ----+
static -------------+-> grid_predict -> assemble
```

It contains no observations, station features, model training, or evaluation.
Do not force `--stages evaluate` when requesting observation-free operation.

For several daily forecasts, extend `models.prediction.end` and `run.end`.
Keep enough history before the first prediction date. Each prediction day
gets a separate midnight issuance and domain product.

### 11.3 Model compatibility

`model_source` points to:

```text
TRAINING_RUN/products/models/
  DOMAIN/
    lstm.keras
    scalers.joblib
    lstm_metadata.json
    xgboost.json
    xgb_metadata.json
```

All required domains must exist. Keep original training periods, feature order,
model hyperparameters, domain bounds/resolution, source field definitions,
coordinate/time names, timing mode, cycle hour, units, and distance reference.
The loader checks those recorded contracts and fails on mismatches.

Source paths, cycle-day offsets, lead lists, tile size, prediction cell-batch
size, and scheduler settings can change without those artifact checks rejecting
them; this is **not** proof that a changed data selection is scientifically
equivalent. Preserve valid-time coverage and review every source-selection
change. Legacy HDF5 models/scaler files are not drop-in substitutes.

## 12. Configure Slurm resources and monitor jobs

### 12.1 Scheduler settings

```yaml
slurm:
  account: arl
  partition: u1-compute
  qos: batch
  python: /shared/path/envs/dafcom/bin/python
  setup: []
  max_array_size: 1000
  max_concurrent: 16
  max_dependency_jobs: 100
  resources: {cpus: 4, memory: 16G, walltime: "02:00:00"}
  stage_resources:
    train_lstm: {cpus: 16, memory: 64G, walltime: "12:00:00"}
    train_xgb: {cpus: 16, memory: 32G, walltime: "06:00:00"}
    grid_predict: {cpus: 4, memory: 16G, walltime: "04:00:00"}
```

`arl`, `u1-compute`, and `batch` are examples inherited from the Ursa scripts,
not guaranteed current allocations. Confirm access and policy with the site.

| Setting | Effect |
|---|---|
| `python` | One executable/path, not a shell command with arguments. Prefer an absolute path. |
| `setup` | Trusted single-line shell commands before worker launch; no automatic activation. |
| `resources` | Default per-task CPUs, memory, and walltime. |
| `stage_resources` | Complete resource object for each named override; omitted fields use model defaults, not inherited custom values from `resources`. |
| `max_array_size` | Maximum tasks in one generated array; must respect the site's array-index limit. |
| `max_concurrent` | `%N` array throttle; chunks are chained so it remains a per-stage cap. |
| `max_dependency_jobs` | Maximum dependency fan-in before lightweight join jobs are introduced; at least 2. |

Use valid stage keys: `ingest`, `static`, `regrid`, `observations`, `features`,
`train_lstm`, `site_lstm`, `train_xgb`, `site_xgb`, `grid_predict`, `assemble`,
and `evaluate`.

If modules are needed, put the site's actual commands in `setup`; they execute
under `set -euo pipefail`. Do not add credentials. The submitting shell must
also have a working DAFCOM environment and `sbatch` on its path.

Generated jobs request one node and one task. CPU settings limit numerical
library threads; no GPU resources are requested by the current generator.
Memory is per task/job, not a workflow-wide pool.

Independent stages can run concurrently. `max_concurrent: 16` is not a global
cap of 16 jobs across the workflow. Budget filesystem contention as well as CPUs.

### 12.2 Inspect task scale before submitting

The reference January-May CONUS run has 240 tiles at 0.01 degrees:

- 151 ingest tasks;
- 1,440 static tasks, including fixed maps and five monthly climatologies;
- 36,240 daily regrid tasks;
- 7,440 grid prediction tasks if maps are enabled.

These are generated tasks, not duplicated source files. With the reference
array limit, the complete graph creates 56 batch scripts before any join jobs.
Different domains, dates, stages, or tile sizes change these counts.

The `features` stage samples tiles but works at day/domain granularity, and
training reads station tables; not every stage is bounded by the grid-cell
prediction batch size. Measure each stage's memory on representative data.
Full CONUS retained native/regridded products can consume substantial scratch
storage. No automatic retention cleanup is implemented.

### 12.3 Monitor

Submission returns job IDs and records them immediately in the workflow's
`submission.json`. Use standard site tools:

```bash
squeue -u "$USER"
sacct -j 123456 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
scontrol show job 123456
```

Replace `123456` with a recorded job ID. Some accounting fields may require
site-specific options.

Inspect the run's `logs/` directory. Generated output names include stage,
chunk, array job ID, and task ID, for example:

```text
logs/regrid-0-123456_7.out
logs/regrid-0-123456_7.err
```

`afterok` means a failed upstream array prevents dependents from starting.
`DependencyNeverSatisfied` is an upstream failure symptom, not a reason to
remove dependencies blindly.

## 13. Understand outputs

Every prepared workflow contains:

```text
RUN_DIRECTORY/
  config.yaml               # Resolved configuration snapshot
  manifest.json             # Stage/chunk paths, counts, and checksums
  submission.json           # Created on submission; accepted jobs and status
  logs/
  STAGE/
    0.json                  # Task descriptions
    0.slurm                 # Generated array script
  products/
```

Principal products:

| Relative path under `products/` | Content |
|---|---|
| `native/DATE/VARIABLE.nc` | Source-grid fields with validated hourly times. |
| `static/DOMAIN/[MONTH/]TILE/VARIABLE.nc` | Fixed or monthly regridded maps. |
| `regrid/DOMAIN/DATE/TILE/VARIABLE.nc` | Daily predictors with tile halos. |
| `observations/DOMAIN/DATE/observations.csv` | Filtered hourly station observations. |
| `features/DOMAIN/DATE/features.csv` | Sampled predictors, derived features, and observations. |
| `models/DOMAIN/` | Models, scalers, metadata, training history, optional search results. |
| `site_lstm/DOMAIN/DATE/site_lstm.csv` | First-hour station predictions used for stacking. |
| `site_lstm_forecast/DOMAIN/DATE/site_lstm_forecast.csv` | All LSTM leads with issuance/valid time and station coordinates. |
| `site_xgb/DOMAIN/DATE/site_xgb.csv` | Matched observations, refined LSTM/XGBoost and UFS values. |
| `grid_predict/DOMAIN/DATE/TILE/grid_predict.nc` | Core tile forecasts, without halo duplication. |
| `forecast/DOMAIN/DATE/forecast.nc` | Assembled domain NetCDF. |
| `evaluation/DOMAIN/evaluation.json` | Pooled paired statistics. |
| `evaluation/DOMAIN/evaluation.png` | Scatter diagnostics. |

`DATE` is `YYYY-MM-DD`; monthly directories use the month's first date.

Inspect a completed forecast:

```bash
python - <<'PY'
import xarray as xr

path = "/shared/scratch/dafcom/forecast-20260605/products/forecast/conus/2026-06-05/forecast.nc"
with xr.open_dataset(path) as dataset:
    print(dataset)
    print(dataset.time.values)
    print(dataset.lstm_pm25.shape)
    print(dataset.xgb_predictions_pm25.shape)
PY
```

With the reference CONUS grid, shapes are `(24, 2400, 6000)` for LSTM and
`(2400, 6000)` for first-hour XGBoost. Other grids/horizons derive their shapes
from settings. Assembly streams tiles into one domain file; it does not load
the complete domain into RAM.

Internal products are published by atomic rename. A completed file is not
replaced by a partially written one, but that does not make an incomplete run
complete. Check scheduler state, expected products, valid times, and finite
values before delivery.

## 14. Troubleshooting and recovery

| Symptom | Likely cause and action |
|---|---|
| Unknown settings/invalid feature names | Compare YAML keys against the reference; lists must be unique and point to configured or derived features. |
| Expected source file(s), found zero | Check resolved paths, actual filename suffixes, cycle/lead formatting, archive staging, and previous-day availability. |
| Source requires decoded valid-time coordinates | Correct the configured `time` name or source metadata/adapter. Do not silently relabel production data. |
| Require exactly 24 unique hourly valid times | Check cycle offsets, lead selection, `time_slice`, overlaps, and day-boundary coverage. |
| Configure indexers for a surface dimension | Set the real vertical dimension to the intended surface index. |
| xESMF/ESMF import failure | Ensure esmpy and xESMF are importable by the exact worker Python on compute nodes. |
| Nonfinite samples/features | Check domain/source overlap, missing inputs, unmapped regrid cells, units, and source quality; no silent filling is provided. |
| No contiguous histories / too few training windows | Check station IDs, hourly gaps, observation filtering, interval length, lookback, and validation cutoff. |
| Station identity/coordinates changed | Use a stable station ID variable; positional indices may have shifted between monthly files. |
| Model configuration/scientific contract mismatch | Start from the training snapshot and restore compatible settings, or retrain intentionally. Do not edit artifact metadata to bypass checks. |
| Invalid account, partition, or QoS | Confirm current Ursa allocation/policy and update scheduler settings before preparing a new run. |
| Out of memory | Inspect `MaxRSS`; reduce tile/cell-batch sizes where applicable or increase stage memory. Training/station stages require separate sizing. |
| Walltime exceeded | Increase the affected stage walltime or reduce workload after measuring a canary. |
| Modified workflow snapshot | A prepared config/task/script changed. Prepare a new run rather than patching checksums. |
| Run directory already exists | Choose a new run ID. Preparation intentionally refuses to overwrite a workflow. |
| Submission ledger already exists | The workflow was already submitted or partly submitted. Inspect recorded jobs before taking action. |
| Dependency never satisfied | Find the failed upstream task in accounting/logs; do not force downstream jobs to run on missing products. |

### Recovery procedure

1. Inspect `submission.json`, Slurm state, and the first upstream failure.
2. If abandoning the run, cancel **only known jobs belonging to that run**:
   `scancel 123456 123457` with the actual recorded IDs.
3. Confirm no old jobs will continue writing or consuming resources.
4. Correct the source configuration, environment, or input data.
5. Prepare a **new** workflow directory and submit it after inspection.

Submission can fail after accepting some jobs. The ledger records accepted IDs;
`complete: true` means submission finished, **not** that computation succeeded.
Failed dependencies may remain pending until an operator cancels them.
Do not delete the ledger and resubmit the same workflow blindly.

The current CLI does not implement resume, completed-product caching, dependency
repair, or automatic cancellation. Manual worker replay is an advanced operation
and is not a substitute for reestablishing a consistent DAG.

## 15. Acceptance checklist

Before a production-sized training or forecast run:

- [ ] Worker Python imports DAFCOM, TensorFlow, XGBoost, NetCDF, and required ESMF/xESMF.
- [ ] Shared paths and permissions work on compute nodes, not just the login node.
- [ ] Source field names, units, surface indices, and valid-time conventions match actual files.
- [ ] Every `run` day has complete unique hourly input coverage, including required previous cycles.
- [ ] Station IDs are stable and observation QC/rounding is appropriate.
- [ ] Fitting periods are disjoint and withheld verification dates are genuinely withheld.
- [ ] `model_source` artifacts match domain, feature order, timing, and model contracts.
- [ ] Plan counts, stage resources, array limits, and scratch capacity have been reviewed.
- [ ] A small canary completes through evaluation or assembled forecast delivery.
- [ ] Metrics, paired counts, tile boundaries, timestamps, and forecast shapes meet agreed thresholds.
- [ ] The actual Slurm jobs finish successfully, and products are checked before downstream use.
- [ ] Software environment, source revision, configuration, and model provenance are retained.

Synthetic package tests support correctness of the implementation; they do not
establish scientific equivalence, forecast skill, or NOAA operational approval.
Full UFS operational integration still requires site validation and an approved
data-availability, launch, monitoring, recovery, and delivery policy.
