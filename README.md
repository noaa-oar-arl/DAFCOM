# NOAA's DynAmic Forecasting of Air Composition using Optimized Machine Learning

DAFCOM combines UFS air-quality predictors, observations, LSTM forecasts, and
XGBoost refinement. The configurable Python package is the supported execution
path on this branch. The legacy forecast and reanalysis script trees have been
removed; their scientific reference implementations remain available in
[Git history](https://github.com/drnimbusrain/DAFCOM/tree/bddff9f9593759bdbe5047e428aa49627d3ce8c0/src).
The package does not depend on those scripts or implement the legacy reanalysis
research workflows.

For detailed configuration, training/evaluation, forecast-only execution, and
Ursa submission/recovery instructions, see the
[DAFCOM user guide](documents/user-guide.md).

## Install and configure

Use Python 3.12 for the tested TensorFlow environment. The core package supports
Python 3.10+; availability of optional ML/ESMF dependencies depends on Python and
platform.

```bash
python -m pip install -e '.[ml,test]'
```

Curvilinear UFS grids additionally require ESMF/esmpy and xESMF in the same
environment. Provision ESMF/esmpy with the site's environment/module or conda
tools, then install `'.[regrid]'`. Pip alone does not provision ESMF/esmpy.

Edit [`configs/ursa.yaml`](configs/ursa.yaml), including source file templates,
coordinate/vertical dimension names, dates, features, domains, grid resolution,
and scheduler resources. These paths are examples of the existing Ursa archive,
not portable defaults or a claim that those files are available.

```bash
dafcom validate configs/ursa.yaml
dafcom plan configs/ursa.yaml
```

All dates are UTC and intervals are **start-inclusive, end-exclusive**.
Resolution is in degrees; `0.01` degrees is not uniformly 1 km.

## Local execution or Ursa submission

```bash
# Input/observation/station-feature preparation only, with prerequisite stages:
dafcom run configs/ursa.yaml --stages features --workers 4

# Prepare the entire training/prediction/evaluation workflow, without submission:
dafcom prepare configs/ursa.yaml --run-dir /shared/scratch/dafcom/run-001

# Inspect the generated config, manifest, and .slurm files before submitting:
dafcom submit --run-dir /shared/scratch/dafcom/run-001
```

Run directories must be new and on a filesystem visible to all workers. They
contain configuration/task snapshots, job-array scripts, per-task logs, a
submission ledger, and isolated `products/`. Account `arl`, partition
`u1-compute`, and QoS `batch` are configurable example defaults; confirm them
with Ursa administrators. Setup commands and the Python executable are explicit.

Arrays have bounded size/concurrency and `afterok` dependencies. Array chunks
within a stage are chained, so the concurrency cap is **per stage**, not multiplied
by the number of chunks. Independent stages may run concurrently.

## Forecast without observations or retraining

In a new configuration:

1. Set `model_source` to a previous run's `products/models/` directory.
2. Keep the trained model settings and training periods; set `run` to the required
   input history plus forecast period and update `models.prediction`.
3. Set `models.site_predictions: false` and `models.grid_predictions: true`.
4. Prepare/submit as above, or use `dafcom run`.

The resulting graph excludes observations, station sampling, training, and
evaluation. It loads compatible trained artifacts, predicts tiles in bounded
cell batches, and assembles one NetCDF per domain/day without loading a complete
domain into RAM.

## Scientific compatibility and validation

This migration requires **retraining**. Source valid times, chronological
holdouts, training-only scalers, disjoint LSTM/XGBoost fitting periods, contiguous
sequences, and properly scaled grid inference intentionally correct legacy
behavior. An explicit `legacy_index` timing option supports historical timing
comparisons; it does not restore all legacy ML behavior.

See the [branch comparison, stage mapping, configuration contract, and
operational acceptance checklist](documents/modernization.md).

```bash
python -m pytest -q
python -m ruff check src/dafcom tests
python -m ruff format --check src/dafcom tests
```

Tests use synthetic data, including actual TensorFlow/XGBoost training when the
ML extra is installed. Real Ursa execution, production UFS/ESMF data, scientific
skill/equivalence, and operational resource sizing require site validation.


### Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
