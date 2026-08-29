"""Shared, task-independent cache for fixed-window model predictions."""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "SEGCLR_PREDICTION_CACHE",
        "/orcd/scratch/orcd/013/jcbliao/segclr/window_prediction_cache",
    )
)
SCHEMA_VERSION = 1
REQUIRED_ARRAYS = {
    "split", "root_id", "center_index", "center_xyz", "prediction", "target",
    "num_embeddings", "schema_version",
}


def cache_path(run_name: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"{run_name}.npz"


def load_prediction_cache(
    run_name: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    required_splits: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Load and validate a cache; split values are ``train`` and ``test``."""
    path = cache_path(run_name, cache_dir)
    with np.load(path) as z:
        missing = REQUIRED_ARRAYS.difference(z.files)
        if missing:
            raise ValueError(f"legacy/incomplete prediction cache {path}: missing {sorted(missing)}")
        arrays = {name: z[name] for name in z.files}
    if int(arrays["schema_version"].reshape(-1)[0]) != SCHEMA_VERSION:
        raise ValueError(f"unsupported prediction-cache schema in {path}")
    n = len(arrays["prediction"])
    for name in ("split", "root_id", "center_index", "center_xyz", "target"):
        if len(arrays[name]) != n:
            raise ValueError(f"{path}: {name} has {len(arrays[name])} rows, expected {n}")
    present = set(arrays["split"].astype(str).tolist())
    absent = set(required_splits).difference(present)
    if absent:
        raise ValueError(f"{path}: missing required split(s) {sorted(absent)}")
    return arrays


def save_prediction_cache(
    run_name: str,
    arrays: dict[str, np.ndarray],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> Path:
    """Atomically publish a shared cache after validating its schema."""
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "format": "one compressed NumPy .npz per model run",
        "row_granularity": "one fixed-embedding window center",
        "arrays": {
            "split": "train or test", "root_id": "uint64 cell/root ID",
            "center_index": "node index in data/graph_cache/<root_id>.pt",
            "center_xyz": "float32 nanometer coordinates",
            "prediction": "finest-level class code", "target": "finest-level class code",
            "num_embeddings": "fixed neighborhood size", "schema_version": "format version",
        },
        "class_names": "data/manifest.json plus data.dataset_lcpn active hierarchy",
    }
    (directory / "schema.json").write_text(json.dumps(schema, indent=2))
    dest = cache_path(run_name, directory)
    payload = dict(arrays)
    payload["schema_version"] = np.array([SCHEMA_VERSION], np.int16)
    tmp = directory / f".{run_name}.{os.getpid()}.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, dest)
    load_prediction_cache(run_name, directory)
    return dest
