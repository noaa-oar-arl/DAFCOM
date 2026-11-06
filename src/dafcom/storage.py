"""Deterministic, isolated outputs with atomic publication."""

from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def product(config, stage, domain="", day="", tile="", variable="", suffix="nc"):
    parts = [stage, domain, str(day), tile]
    directory = config.output_root.joinpath(*(part for part in parts if part))
    return directory / f"{variable or stage}.{suffix}"


@contextmanager
def atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}{path.suffix}")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_netcdf(dataset, path):
    with atomic_path(path) as temporary:
        dataset.to_netcdf(temporary, engine="netcdf4")


def write_csv(frame, path):
    with atomic_path(path) as temporary:
        frame.to_csv(temporary, index=False)
