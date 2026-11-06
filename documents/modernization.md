# Ursa workflow migration

For step-by-step installation, configuration, training, evaluation, and Ursa
forecast operation, see the [DAFCOM user guide](user-guide.md).

## Branch comparison

The original `ursa_2026` tip (`7d63f5b`, before this migration) differs from
`origin/main` by **864 changed files: 863 additions and one deletion**.
The additions are 433 Python files, 406 Slurm files, 19 shell files, and five
other files. Thus, "864 new scripts" describes the scale but not the exact count.
The forecast tree contains 858 Python/Slurm/shell files.

`4c4d5bf` introduced the workflow; `7d63f5b` synchronized tested personal Ursa
scripts. Compared with main's forecast placeholder, this provides substantially
more scientific functionality, but experiments and execution settings are
encoded in directory copies rather than reusable interfaces:

- Input preparation repeats months and variable-specific regridders.
- Interpolation repeats daily loaders and station-interpolation code.
- Shell loops submit existing daily copies; other shell scripts use `sed` to
  change dates in those copies.
- Grid dimensions `488 x 775`, target dimensions `2400 x 6000`, coordinate bounds,
  dates, paths, feature orders, and resource requests are embedded in scripts.
- LSTM site/full-domain/east/west inference duplicates feature loading and
  differs in scaling behavior.

The separate `modern_package` branch already introduced YAML, loaders,
preprocessing, model classes, tests, and an entry script. Its configuration still
contains experiment-specific dates, source dimensions, and paths; its
`pyproject.toml` configures linting rather than installable project metadata.
Its dataflow/model contracts differ from the tested Ursa scripts. This migration
uses the modular/configuration direction without blindly merging incompatible
scientific implementations or introducing a second regridding backend.

The active implementation is now [`src/dafcom`](../src/dafcom/). The legacy
forecast and reanalysis trees have been removed from this branch to keep the
checkout focused on the package. Neither tree is a package dependency.
Reference scripts for real-data comparisons remain available in the
[pre-removal source snapshot](https://github.com/drnimbusrain/DAFCOM/tree/bddff9f9593759bdbe5047e428aa49627d3ce8c0/src).
Removing those trees does not establish scientific equivalence or operational
acceptance, and the package does not replace the legacy reanalysis research
workflows.

## Consolidated stage mapping

| Legacy stage | Package stage | Work unit |
|---|---|---|
| 1: meteo/AOD/pollutant BEV and surface maps | `ingest` | UTC day, all dynamic variables |
| 1: variable/month regridding | `regrid` | day/domain/tile, all dynamic variables |
| static inputs/emissions read repeatedly | `static` | domain/tile; monthly for climatologies |
| 2: monthly AirNow extraction | `observations` | day/domain |
| 3: model input imports and station interpolation | `features` | day/domain, vectorized tile sampling |
| 4: merge/clean and LSTM fitting | `train_lstm` | domain |
| 5, 6.1, 7.2: LSTM station application | `site_lstm` | day/domain |
| 6.2-3: stacking merge and XGBoost fitting | `train_xgb` | domain |
| 7.3-4: stacking merge and XGBoost application | `site_xgb` | day/domain |
| 5: full/east/west map preparation and prediction | `grid_predict` | day/domain/tile |
| map collection | `assemble` | day/domain, streamed NetCDF writes |
| 7.5-6 and 8: evaluation/bias-correction comparison | `evaluate` | domain |

Merging/cleaning is integrated with feature tables, unique station/hour keys,
positive finite observations, and sequence construction. Evaluation writes
JSON metrics and scatter diagnostics for XGBoost, LSTM, UFS, and optionally an
external bias-correction variable. Configure that variable as a dynamic input
and set `baseline_variable`; it is sampled on exactly the same stations/hours.
The many experiment-specific legacy plot variants are not separate package
stages.

## Configuration and data contracts

[`configs/ursa.yaml`](../configs/ursa.yaml) is the authoritative example.
Unknown settings, invalid dates/bounds/feature names, duplicate features, and
nonintegral grid shapes fail validation.

- **Paths:** relative paths resolve against the input YAML, not the submission
  working directory. Templates support `{date}` (`YYYYMMDD`), `{year}`, `{month}`,
  `{day}`, `{month_start}`, `{next_month}`, `{cycle:02d}`, `{variable}`, and
  `{lead:03d}`. No environment-variable or arbitrary expression evaluation occurs.
  Quote YAML chemical names such as `"no"` because YAML 1.1 treats bare `no` as
  a boolean.
- **Dynamic fields:** each variable declares source path(s), input field names,
  optional summed fields, coordinate/time names, surface `indexers`, and units.
  Extra nonsingleton vertical dimensions require explicit `indexers`.
  `time_slice: [start, end]` selects a multi-time source's forecast segment.
  Sources sharing filenames are opened together during ingestion.
- **Time:** source mode requires decoded datetime valid-time coordinates and
  exactly 24 unique hourly records per day. Previous/current cycle offsets
  allow 12Z archive files to cover UTC midnight-to-midnight. Missing, duplicate,
  or misaligned hours are errors, never silently relabeled or filled.
  Inputs for the previous cycle must exist even on the first configured day.
- **Historical timing:** `time_alignment: legacy_index`,
  `cycle_day_offsets: [0]`, and explicitly ordered `leads` relabel selected
  records by index. This is only for comparisons with the old midnight-based
  convention. Do not mix source and legacy timing unintentionally.
- **Static fields:** `static_period: fixed` regrids once per domain/tile;
  `monthly` regrids once per calendar month/domain/tile, preserving monthly
  emissions across year boundaries. Static arrays are broadcast, not copied
  once per hour. Historical reference years are settings in source templates.
- **Grids:** latitude is descending from north, longitude ascending from west,
  with south/east endpoints excluded. Dimensions derive from bounds/resolution.
  Longitude normalization accepts both 0..360 and -180..180 sources. Dateline
  crossing and moving grids are explicitly unsupported.
- **Regridding:** rectilinear grids use xarray linear/nearest interpolation;
  curvilinear UFS uses xESMF bilinear/nearest-s2d and requires ESMF/esmpy.
  Regridding weights are shared by variables with identical source/target grids
  and method within a task. Temporary weight filenames are isolated per process.
  Categorical land use uses configurable nearest-neighbor regridding.
- **Tiles:** integer cell bounds define exact, nonoverlapping output ownership.
  Regridded tiles contain a one-cell halo for interpolation at tile boundaries.
  Station sampling does not extrapolate beyond actual grid coordinates.
  Gridded inference outputs only core cells; assembly verifies coordinates/time
  before writing each tile into its assigned hyperslab.
  See the [domain tiling and memory explanation](user-guide.md#51-how-a-domain-becomes-independently-executable-tiles)
  for the CONUS partition example, cell-batch memory estimates, and trade-offs.
- **Observations:** NetCDF, CSV, or XLSX inputs are supported (XLSX additionally
  needs the `excel` extra). Missing sentinels are masked before
  hourly means. NetCDF station IDs should be configured using `site`; positional
  IDs are only suitable when the source station ordering is stable. Cross-file
  identity/coordinate changes fail when tables are merged. String IDs retain
  leading zeros. Empty daily samples are logged; an empty training/stacking
  dataset is an error.
- **Features:** configured names and order are shared by training/site/grid
  paths. The example retains the legacy 19 environmental plus hour/day LSTM
  features. Coordinate/distance features are available explicitly; distances
  are vectorized haversine values at locations relative to configured global
  anchors, not repeated full-domain hourly matrices.

Native outputs carry actual time coordinates and source filenames. Internal
tables are CSV rather than intermediate XLSX copies. Products are published by
atomic rename; interrupted writes do not replace a completed product.

## Intentional ML corrections

These are not bitwise-compatible replacements for existing trained models.
Retraining and scientific review are required:

1. LSTM inputs contain `lookback_hours` contiguous hours from the same station;
   targets are the following `forecast_hours`. Every internal timestamp is
   checked, not just endpoints.
2. Validation is chronological across all stations. Training windows end before
   the cutoff; validation windows start after it, purging overlapping histories.
   RobustScalers fit only pre-cutoff rows.
3. XGBoost training follows the LSTM training period; prediction follows both.
   The example trains LSTM on Jan-Mar, XGBoost on April, predicts May. This
   intentionally replaces legacy Jan-Apr fitting for both stacked models.
4. Optional randomized XGBoost search uses chronological folds grouped by valid
   time, never splitting stations at the same hour across a fold boundary.
   The configurable `models.xgboost.scoring` selects the search metric
   (default `r2`); negative error scorers support RMSE/MAE selection.
   Search parallelism is one, with estimator threads limited to allocated CPUs.
5. Both site and grid inference transform predictors with the fitted X scaler
   and inverse-transform predictions with the Y scaler. The legacy small-domain
   script loaded scalers but predicted directly from unscaled inputs.
6. LSTM station forecast tables retain issuance, lead, and valid time for every
   horizon. The first forecast hour, keyed by station/valid time, is used for
   XGBoost stacking, matching the refinement concept in legacy stages 6/7.
   Standalone gridded LSTM output covers the full horizon.
7. XGBoost grid output is **only the first valid forecast hour**, consistent
   with first-hour stacking training. Extending refinement to all leads requires
   lead-aware training; the package does not silently apply that model to a
   different lead distribution.
8. Artifacts include scalers, model settings, training periods, and scientific
   input/grid contracts. Mismatches fail rather than loading arbitrary HDF5 or
   joblib artifacts from the old workflow.

The target remains PM2.5. Additional UFS chemical variables can be ingested and
selected as predictors; generalizing the observation target and model semantics
to other species requires a separate scientifically validated extension.

## Execution and operational integration

`plan` is read-only. `prepare` creates a new run directory and snapshots absolute
paths/configuration, task lists, generated Slurm scripts, and SHA-256 checksums.
It does not submit jobs. Each workflow has isolated products, so different
configurations/dates/subdomains cannot overwrite another workflow's files.

`submit` verifies snapshots, uses `sbatch --parsable`, records every accepted
job ID immediately, and wires stage-level `afterok` dependencies. Failed
upstream arrays block dependent stages. Long stages split at `max_array_size`;
chunks are chained so `max_concurrent` applies to the whole stage. Independent
stages can overlap, so it is not a workflow-wide CPU/job cap. Domain/tile work is
interleaved within stages. Dependency fan-in is bounded with lightweight joins.
Stdout/stderr names include stage, chunk, array job, and task IDs.

The submission ledger prevents duplicate submission, including after a partial
failure. After such a failure, inspect recorded jobs and cancel applicable jobs
explicitly with `scancel JOB_ID` before preparing a new workflow. There is no
automatic cancellation, automatic dependency rewriting, or automatic retry.
Failed Slurm dependencies can remain pending and require operator cleanup.
Workers verify config/task checksums before execution. These are reproducibility
guards, not authentication or a substitute for shared-filesystem permissions.

Local execution uses the same tasks, stage barriers, and a process pool (spawn,
not threads, to isolate NetCDF/HDF5/TensorFlow). `--workers` controls concurrent
processes; allow `workers * cpus_per_task` CPUs. Each task bounds numerical-library
threads. This initial scheduler uses stage barriers rather than per-day
pipelining, favoring a transparent, testable DAG over maximum overlap.

For forecast-only operation, set `model_source` to a previous run's
`products/models`, disable `site_predictions`, and provide only the input history
and forecast period in `run`. Keep the original trained model settings/training
periods; they may now lie outside `run`. The selected graph is:

```text
ingest -> regrid ----+
                    +-> grid_predict -> assemble
static -------------+
```

No observations are needed in this mode. Model artifacts are read-only and
products go to the new run. Domain names and scientific grid/feature contracts
must match trained artifacts; changing bounds/resolution or timing requires
retraining, rather than silently transferring a model.

Principal outputs under each workflow's `products/`:

```text
native/YYYY-MM-DD/VARIABLE.nc
static/DOMAIN/[YYYY-MM-01/]TILE/VARIABLE.nc
regrid/DOMAIN/YYYY-MM-DD/TILE/VARIABLE.nc
observations/DOMAIN/YYYY-MM-DD/observations.csv
features/DOMAIN/YYYY-MM-DD/features.csv
models/DOMAIN/{lstm.keras,scalers.joblib,xgboost.json,...}
site_lstm/DOMAIN/YYYY-MM-DD/site_lstm.csv
site_lstm_forecast/DOMAIN/YYYY-MM-DD/site_lstm_forecast.csv
site_xgb/DOMAIN/YYYY-MM-DD/site_xgb.csv
grid_predict/DOMAIN/YYYY-MM-DD/TILE/grid_predict.nc
forecast/DOMAIN/YYYY-MM-DD/forecast.nc
evaluation/DOMAIN/evaluation.{json,png}
```

At 0.01 degrees CONUS has 14.4 million cells and 240 tiles at the example
256-cell tile size. The Jan-May example plans 36,240 daily regrid tasks and 7,440
grid-prediction tasks, not one Python/Slurm copy per task. Tiling bounds RAM, not
total storage or runtime. Size scratch quotas, walltime, native/tile retention,
and tile/batch dimensions on representative files before a full run.

## Acceptance before NOAA UFS operational use

Automated coverage includes analytical interpolation, coordinate boundaries and
halos, calendar rollover, missing/duplicate hours, observations, sequence
continuity, scaling, same-hour CV isolation, scheduler limits/failures, process
parallelism, and real synthetic LSTM/XGBoost training/site/grid prediction.
The synthetic forecast-only run must reproduce the trained run's gridded output.

Still required on Ursa:

- Verify actual archive filenames, valid-time/cycle conventions, vertical
  dimension selectors, variable units, static grid coordinates, station IDs,
  and ESMF/xESMF installation against production data.
- Compare regridded fields and station features against trusted legacy outputs,
  separating time-alignment corrections from interpolation/model changes.
- Retrain and evaluate withheld periods/stations, extrema and missing-data
  behavior; agree scientific skill/error thresholds and model acceptance policy.
- Execute a small Slurm canary using the actual account/partition/QoS and shared
  Python environment; measure memory, storage, filesystem contention, and runtime.
- Establish pinned site environments, model registry/versioning, data-availability
  checks, scheduler recovery, retention policy, and operational monitoring.
- Integrate validated forecast-only tasks/products with the selected UFS
  workflow manager and approved production NetCDF/metadata delivery contract.

This is an executable modernization foundation, not a declaration of operational
readiness or proof of scientific equivalence on NOAA production datasets.
