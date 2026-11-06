"""Dependency-aware local execution and immutable Ursa job-array submission."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import re
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import load_config, save_config
from .grid import Tile, tiles
from .storage import atomic_path

DEPENDENCIES = {
    "ingest": (),
    "static": (),
    "regrid": ("ingest",),
    "observations": (),
    "features": ("regrid", "static", "observations"),
    "train_lstm": ("features",),
    "site_lstm": ("features", "train_lstm"),
    "train_xgb": ("site_lstm",),
    "site_xgb": ("site_lstm", "train_xgb"),
    "grid_predict": ("train_xgb", "regrid", "static"),
    "assemble": ("grid_predict",),
    "evaluate": ("site_xgb",),
}


def dependency_graph(config=None):
    if config is None or config.model_source is None:
        return DEPENDENCIES
    training = {"train_lstm", "train_xgb"}
    return {
        stage: tuple(parent for parent in parents if parent not in training)
        for stage, parents in DEPENDENCIES.items()
        if stage not in training
    }


def selected_stages(stages=None, config=None):
    graph = dependency_graph(config)
    required = set(graph if stages is None else stages)
    unknown = required - graph.keys()
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    pending = list(required)
    while pending:
        for dependency in graph[pending.pop()]:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return [stage for stage in graph if stage in required]


def plan(config, stages=None):
    unknown = config.slurm.stage_resources.keys() - DEPENDENCIES.keys()
    if unknown:
        raise ValueError(f"Unknown stage_resources: {sorted(unknown)}")
    if stages is None:
        stages = []
        if config.models.site_predictions:
            stages.append("evaluate")
        if config.models.grid_predictions:
            stages.append("assemble" if config.models.assemble_grid else "grid_predict")
        if config.model_source is None:
            stages.append("train_xgb")
    tasks = {stage: [] for stage in selected_stages(stages, config)}

    def add(stage, **settings):
        if stage in tasks:
            tasks[stage].append({"stage": stage, **settings})

    for day in config.run.days():
        add("ingest", day=day.isoformat())
    for domain, grid in config.domains.items():
        domain_tiles = tiles(grid)
        fixed = [
            name
            for name, spec in config.variables.items()
            if spec.kind == "static" and spec.static_period == "fixed"
        ]
        monthly = [
            name
            for name, spec in config.variables.items()
            if spec.kind == "static" and spec.static_period == "monthly"
        ]
        if fixed:
            for tile in domain_tiles:
                add(
                    "static",
                    domain=domain,
                    day=config.run.start.isoformat(),
                    tile=vars(tile),
                    variables=fixed,
                )
        if monthly:
            months = sorted({day.replace(day=1) for day in config.run.days()})
            for month in months:
                for tile in domain_tiles:
                    add(
                        "static",
                        domain=domain,
                        day=month.isoformat(),
                        tile=vars(tile),
                        variables=monthly,
                    )
        for day in config.run.days():
            for tile in domain_tiles:
                add("regrid", day=day.isoformat(), domain=domain, tile=vars(tile))
            add("observations", day=day.isoformat(), domain=domain)
            add("features", day=day.isoformat(), domain=domain)
        add("train_lstm", domain=domain)
        periods = (
            (config.models.prediction,)
            if config.model_source is not None
            else (config.models.xgb_training, config.models.prediction)
        )
        for period in periods:
            for day in period.days():
                add("site_lstm", day=day.isoformat(), domain=domain)
        add("train_xgb", domain=domain)
        for day in config.models.prediction.days():
            add("site_xgb", day=day.isoformat(), domain=domain)
            if config.models.grid_predictions:
                for tile in domain_tiles:
                    add("grid_predict", day=day.isoformat(), domain=domain, tile=vars(tile))
                if config.models.assemble_grid:
                    add("assemble", day=day.isoformat(), domain=domain)
        add("evaluate", domain=domain)
    for items in tasks.values():
        items.sort(
            key=lambda item: (
                item.get("day", ""),
                item.get("tile", {}).get("y0", 0),
                item.get("tile", {}).get("x0", 0),
                item.get("domain", ""),
            )
        )
    return tasks


def run_task(config, task):
    from threadpoolctl import threadpool_limits

    from . import assemble, evaluate, features, inputs, models

    stage = task["stage"]
    cpus = config.slurm.stage_resources.get(stage, config.slurm.resources).cpus
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ):
        os.environ[name] = str(cpus)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    day = date.fromisoformat(task["day"]) if "day" in task else None
    domain = task.get("domain")
    tile = Tile(**task["tile"]) if "tile" in task else None
    logging.info("Starting task %s", task)
    with threadpool_limits(limits=cpus):
        if stage == "ingest":
            inputs.ingest_many(
                config,
                day,
                [variable for variable, spec in config.variables.items() if spec.kind == "dynamic"],
            )
        elif stage in ("static", "regrid"):
            regridders = {}
            for variable, spec in config.variables.items():
                if (spec.kind == "static") == (stage == "static") and (
                    stage != "static" or variable in task["variables"]
                ):
                    inputs.regrid(config, day, domain, tile, variable, regridders)
        elif stage == "observations":
            inputs.observations(config, day, domain)
        elif stage == "features":
            features.sample(config, day, domain)
        elif stage == "train_lstm":
            models.train_lstm(config, domain)
        elif stage == "site_lstm":
            models.site_lstm(config, day, domain)
        elif stage == "train_xgb":
            models.train_xgb(config, domain)
        elif stage == "site_xgb":
            models.site_xgb(config, day, domain)
        elif stage == "grid_predict":
            models.grid_predict(config, day, domain, tile)
        elif stage == "assemble":
            assemble.assemble(config, day, domain)
        elif stage == "evaluate":
            evaluate.evaluate(config, domain)
        else:
            raise ValueError(f"Unknown task stage: {stage}")
    logging.info("Completed task %s", task)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script(config, run_dir, stage, chunk, count):
    resources = config.slurm.stage_resources.get(stage, config.slurm.resources)
    command = shlex.join(
        [
            config.slurm.python,
            "-m",
            "dafcom",
            "worker",
            "--run-dir",
            str(run_dir),
            "--stage",
            stage,
            "--chunk",
            str(chunk),
        ]
    )
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=dafcom-{stage}",
            f"#SBATCH --account={config.slurm.account}",
            f"#SBATCH --partition={config.slurm.partition}",
            f"#SBATCH --qos={config.slurm.qos}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={resources.cpus}",
            f"#SBATCH --mem={resources.memory}",
            f"#SBATCH --time={resources.walltime}",
            f"#SBATCH --array=0-{count - 1}%{config.slurm.max_concurrent}",
            "set -euo pipefail",
            *config.slurm.setup,
            f"cd {shlex.quote(str(run_dir))}",
            'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"',
            'export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"',
            'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"',
            command + ' --index "${SLURM_ARRAY_TASK_ID}"',
            "",
        ]
    )


def prepare(config, run_dir=None, stages=None):
    tasks = plan(config, stages)
    if run_dir is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        run_dir = config.output_root / "workflows" / name
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    config = config.model_copy(deep=True)
    config.output_root = run_dir / "products"
    save_config(config, run_dir / "config.yaml")
    manifest = {"schema_version": 1, "config_sha256": digest(run_dir / "config.yaml"), "stages": {}}
    for stage, items in tasks.items():
        entries = []
        directory = run_dir / stage
        directory.mkdir()
        for chunk, offset in enumerate(range(0, len(items), config.slurm.max_array_size)):
            batch = items[offset : offset + config.slurm.max_array_size]
            task_file = directory / f"{chunk}.json"
            task_file.write_text(json.dumps(batch, indent=2), encoding="utf-8")
            batch_file = directory / f"{chunk}.slurm"
            batch_file.write_text(
                script(config, run_dir, stage, chunk, len(batch)), encoding="utf-8"
            )
            entries.append(
                {
                    "tasks": str(task_file.relative_to(run_dir)),
                    "tasks_sha256": digest(task_file),
                    "script": str(batch_file.relative_to(run_dir)),
                    "script_sha256": digest(batch_file),
                    "count": len(batch),
                }
            )
        manifest["stages"][stage] = entries
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def verified_manifest(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or digest(run_dir / "config.yaml") != manifest["config_sha256"]
    ):
        raise ValueError("Workflow configuration was modified after preparation")
    return run_dir, manifest


def verified_file(run_dir, entry, kind):
    path = (run_dir / entry[kind]).resolve()
    if not path.is_relative_to(run_dir) or digest(path) != entry[f"{kind}_sha256"]:
        raise ValueError(f"Workflow {kind} was modified after preparation")
    return path


def worker(run_dir, stage, chunk, index):
    run_dir, manifest = verified_manifest(run_dir)
    if stage not in manifest["stages"] or not 0 <= chunk < len(manifest["stages"][stage]):
        raise ValueError("Invalid worker stage/chunk")
    entry = manifest["stages"][stage][chunk]
    if not 0 <= index < entry["count"]:
        raise ValueError("Array task index outside manifest")
    tasks = json.loads(verified_file(run_dir, entry, "tasks").read_text())
    run_task(load_config(run_dir / "config.yaml"), tasks[index])


def local_run(run_dir, workers=1):
    run_dir, manifest = verified_manifest(run_dir)
    config = load_config(run_dir / "config.yaml")
    if workers < 1:
        raise ValueError("workers must be positive")
    for stage, entries in manifest["stages"].items():
        tasks = []
        for entry in entries:
            tasks.extend(json.loads(verified_file(run_dir, entry, "tasks").read_text()))
        # TensorFlow thread pools cannot be resized after initialization.
        isolated = stage in {"train_lstm", "site_lstm", "train_xgb", "site_xgb", "grid_predict"}
        if workers == 1 and not isolated:
            for task in tasks:
                run_task(config, task)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [pool.submit(run_task, config, task) for task in tasks]
                for future in futures:
                    future.result()


def submit(run_dir, sbatch="sbatch"):
    run_dir, manifest = verified_manifest(run_dir)
    config = load_config(run_dir / "config.yaml")
    for entries in manifest["stages"].values():
        for entry in entries:
            verified_file(run_dir, entry, "script")
            verified_file(run_dir, entry, "tasks")
    state_path = run_dir / "submission.json"
    # Exclusive creation prevents accidental double submission, including after partial failures.
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump({"jobs": [], "complete": False}, stream)
    jobs = []
    latest = {}

    def persist(complete=False):
        with atomic_path(state_path) as temporary:
            temporary.write_text(json.dumps({"jobs": jobs, "complete": complete}, indent=2))

    def launch(args, stage, chunk):
        result = subprocess.run([sbatch, "--parsable", *args], text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"sbatch failed for {stage}/{chunk}: {result.stderr.strip()}")
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
        if not match:
            raise RuntimeError(f"Unrecognized sbatch job ID for {stage}/{chunk}: {result.stdout!r}")
        job = match[1]
        jobs.append({"stage": stage, "chunk": chunk, "job_id": job})
        persist()
        return job

    def bounded_dependencies(dependencies):
        limit = config.slurm.max_dependency_jobs
        if limit < 2 and len(dependencies) > 1:
            raise ValueError("max_dependency_jobs must be >= 2 for multi-parent stages")
        while len(dependencies) > limit:
            joined = []
            for offset in range(0, len(dependencies), limit):
                group = dependencies[offset : offset + limit]
                joined.append(
                    launch(
                        [
                            "--dependency=afterok:" + ":".join(group),
                            "--account=" + config.slurm.account,
                            "--partition=" + config.slurm.partition,
                            "--qos=" + config.slurm.qos,
                            "--cpus-per-task=1",
                            "--mem=1G",
                            "--time=00:01:00",
                            "--output=" + str(run_dir / "logs" / "join-%j.out"),
                            "--error=" + str(run_dir / "logs" / "join-%j.err"),
                            "--wrap=true",
                        ],
                        "join",
                        len(jobs),
                    )
                )
            dependencies = joined
        return dependencies

    for stage, entries in manifest["stages"].items():
        dependencies = [
            latest[parent] for parent in dependency_graph(config)[stage] if parent in latest
        ]
        dependencies = bounded_dependencies(list(dict.fromkeys(dependencies)))
        previous = None
        for chunk, entry in enumerate(entries):
            parents = [previous] if previous else dependencies
            args = [
                "--output=" + str(run_dir / "logs" / f"{stage}-{chunk}-%A_%a.out"),
                "--error=" + str(run_dir / "logs" / f"{stage}-{chunk}-%A_%a.err"),
                "--chdir=" + str(run_dir),
            ]
            if parents:
                args.append("--dependency=afterok:" + ":".join(parents))
            args.append(str(run_dir / entry["script"]))
            previous = launch(args, stage, chunk)
        if previous:
            latest[stage] = previous
    persist(complete=True)
    return jobs
