import json
from types import SimpleNamespace

import pytest

from dafcom import workflow
from dafcom.config import load_config


def test_selection_and_isolation(config, tmp_path):
    stages = workflow.selected_stages(["site_xgb"])
    assert "ingest" in stages and "train_xgb" in stages and "grid_predict" not in stages
    first = workflow.prepare(config, tmp_path / "first", ["features"])
    second = workflow.prepare(config, tmp_path / "second", ["features"])
    assert (
        load_config(first / "config.yaml").output_root
        != load_config(second / "config.yaml").output_root
    )
    with pytest.raises(FileExistsError):
        workflow.prepare(config, first)
    with pytest.raises(ValueError, match="Unknown"):
        workflow.selected_stages(["missing"])


def test_arrays_and_dependency_submission(config, tmp_path, monkeypatch):
    config.slurm.max_array_size = 2
    config.slurm.max_dependency_jobs = 2
    config.slurm.python = "/an environment/python"
    directory = workflow.prepare(config, tmp_path / "with spaces", ["features"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=f"{100 + len(calls)};ursa\n", stderr="")

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    jobs = workflow.submit(directory)
    assert jobs
    manifest = json.loads((directory / "manifest.json").read_text())
    for entries in manifest["stages"].values():
        for entry in entries:
            assert entry["count"] <= 2
            script = (directory / entry["script"]).read_text()
            assert "#SBATCH --array=0-1%2" in script or "#SBATCH --array=0-0%2" in script
            assert "set -euo pipefail" in script
            assert "'/an environment/python'" in script
    ingest_jobs = [job for job in jobs if job["stage"] == "ingest"]
    assert len(ingest_jobs) == 6
    second_ingest = next(call for call in calls if call[-1].endswith("/ingest/1.slurm"))
    assert f"--dependency=afterok:{ingest_jobs[0]['job_id']}" in second_ingest
    regrid_first = next(call for call in calls if call[-1].endswith("/regrid/0.slurm"))
    assert f"--dependency=afterok:{ingest_jobs[-1]['job_id']}" in regrid_first
    assert any(job["stage"] == "join" for job in jobs)
    assert json.loads((directory / "submission.json").read_text())["complete"]
    with pytest.raises(FileExistsError):
        workflow.submit(directory)


def test_snapshot_changes_block_execution(config, tmp_path):
    directory = workflow.prepare(config, tmp_path / "immutable", ["ingest"])
    (directory / "config.yaml").write_text("modified")
    with pytest.raises(ValueError, match="modified"):
        workflow.worker(directory, "ingest", 0, 0)


def test_task_modification_and_index_validation(config, tmp_path):
    directory = workflow.prepare(config, tmp_path / "tasks", ["ingest"])
    with pytest.raises(ValueError, match="index"):
        workflow.worker(directory, "ingest", 0, -1)
    (directory / "ingest/0.json").write_text("[]")
    with pytest.raises(ValueError, match="modified"):
        workflow.worker(directory, "ingest", 0, 0)


def test_submission_failure_is_recorded(config, tmp_path, monkeypatch):
    directory = workflow.prepare(config, tmp_path / "failure", ["features"])
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="123\n", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="quota exceeded"),
        ]
    )
    monkeypatch.setattr(workflow.subprocess, "run", lambda *args, **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="quota exceeded"):
        workflow.submit(directory)
    state = json.loads((directory / "submission.json").read_text())
    assert state["jobs"][0]["job_id"] == "123"
    assert not state["complete"]


def test_generated_shell_syntax(config, tmp_path):
    import subprocess

    directory = workflow.prepare(config, tmp_path / "shell", ["features"])
    for path in directory.glob("*/*.slurm"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_grid_only_forecast_needs_no_observations_or_training(config, tmp_path):
    config.model_source = tmp_path / "previous" / "models"
    config.models.site_predictions = False
    tasks = workflow.plan(config)
    assert set(tasks) == {"ingest", "static", "regrid", "grid_predict", "assemble"}
    assert len(tasks["grid_predict"]) == 4
    with pytest.raises(ValueError, match="Unknown"):
        workflow.plan(config, ["train_lstm"])


def test_process_parallel_ingestion(synthetic_data, tmp_path):
    directory = workflow.prepare(synthetic_data, tmp_path / "parallel", ["ingest"])
    workflow.local_run(directory, workers=2)
    config = load_config(directory / "config.yaml")
    assert len(list((config.output_root / "native").glob("*/*.nc"))) == 12
