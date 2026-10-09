"""Export predictions on their exact CAVE L2 graph nodes (no spatial join).

This publishes a sibling Neuroglancer precomputed skeleton source named
``skeletons_l2``.  Prediction-cache ``center_index`` values index the same
``data/graph_cache/<root>.pt`` objects used for training and inference, so
assignment is an exact array-index join.  Graph nodes without a model window
remain unavailable (-1); predictions are never propagated to nearby vertices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from export_neuroglancer_predictions import (
    BITS_PER_PREDICTION,
    DEFAULT_OUT,
    PREDICTIONS_PER_ATTRIBUTE,
    PREDICTIONS_PER_SCALAR,
    SCALARS_PER_ATTRIBUTE,
    VertexAttribute,
    _atomic_write,
    _json_bytes,
    attribute_layout,
    completed_runs,
    encode_skeleton,
    load_hierarchy,
    load_manifest,
    load_prediction_cache,
    write_segment_properties,
)
from data.window_prediction_cache import DEFAULT_CACHE_DIR

GRAPH_CACHE = Path(__file__).resolve().parent.parent / "data" / "graph_cache"
SCHEMA_VERSION = 1
SOURCE_NAME = "skeletons_l2"
METADATA_NAME = "prediction_labels_l2.json"
PUBLIC_BASE = (
    "https://g-ffa18e.d1c26e.5898.data.globus.org/microns/"
    "segclr_predictions/fold_test_validation"
)


def grouped_indices(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    arrays = load_prediction_cache(path.stem, path.parent, required_splits=("test",))
    keep = arrays["split"].astype(str) == "test"
    roots = arrays["root_id"][keep].astype(np.uint64)
    centers = arrays["center_index"][keep].astype(np.int64)
    predictions = arrays["prediction"][keep].astype(np.int16)
    targets = arrays["target"][keep].astype(np.int16)
    order = np.argsort(roots, kind="stable")
    roots, centers = roots[order], centers[order]
    predictions, targets = predictions[order], targets[order]
    cuts = np.flatnonzero(np.diff(roots)) + 1
    return {
        int(r[0]): (c, p, t)
        for r, c, p, t in zip(
            np.split(roots, cuts), np.split(centers, cuts),
            np.split(predictions, cuts), np.split(targets, cuts)
        )
    }


def prepare(runs: list[str], out: Path, graph_cache: Path,
            plan_key: str | None) -> None:
    manifest = load_manifest()
    held_out = sorted(
        int(rid) for rid, cell in manifest["cells"].items()
        if cell["split"] == "test" and (graph_cache / f"{rid}.pt").is_file()
    )
    caches = [DEFAULT_CACHE_DIR / f"{run}.npz" for run in runs]
    inputs = []
    for run, cache in zip(runs, caches):
        load_prediction_cache(run, cache.parent, required_splits=("test",))
        stat = cache.stat()
        inputs.append({"path": str(cache), "size": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns})
    body = {
        "schema_version": SCHEMA_VERSION, "runs": runs,
        "attribute_layout": attribute_layout(runs), "cache_inputs": inputs,
        "graph_cache": str(graph_cache), "root_ids": held_out,
        "included_fold": "test/validation", "manifest_split": "test",
        "fold_note": "Held-out whole-cell test fold; validation aliases this fold. No training cells.",
        "join": "prediction center_index -> graph_cache node index (exact; no nearest neighbor)",
    }
    plan_id = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    plan = {**body, "plan_id": plan_id}
    build = out / "builds_l2" / plan_id
    stage = build / SOURCE_NAME
    stage.mkdir(parents=True, exist_ok=True)
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "vertex_attributes": [
            {"id": "target_class", "data_type": "float32", "num_components": 1},
            *[
                {"id": f"prediction_pack_{i:03d}", "data_type": "float32",
                 "num_components": SCALARS_PER_ATTRIBUTE}
                for i in range((len(runs) + PREDICTIONS_PER_ATTRIBUTE - 1)
                               // PREDICTIONS_PER_ATTRIBUTE)
            ],
        ],
    }
    _atomic_write(stage / "info", _json_bytes(info))
    _atomic_write(build / "plan.json", _json_bytes(plan))
    if plan_key:
        _atomic_write(out / "plans_l2" / f"{plan_key}.json", _json_bytes(plan))
    _atomic_write(out / "active_plan_l2.json", _json_bytes(plan))
    print(f"prepared L2 plan {plan_id}: {len(held_out)} roots, {len(runs)} models")


def load_plan(out: Path, plan_key: str | None) -> dict:
    path = out / "plans_l2" / f"{plan_key}.json" if plan_key else out / "active_plan_l2.json"
    plan = json.loads(path.read_text())
    frozen = out / "builds_l2" / plan["plan_id"] / "plan.json"
    if plan.get("schema_version") != SCHEMA_VERSION or json.loads(frozen.read_text()) != plan:
        raise SystemExit(f"invalid L2 export plan: {path}")
    return plan


def export(out: Path, shard: int, shards: int, plan_key: str | None) -> None:
    plan = load_plan(out, plan_key)
    runs = plan["runs"]
    grouped = {}
    for run, expected in zip(runs, plan["cache_inputs"]):
        cache = Path(expected["path"])
        stat = cache.stat()
        if {"path": str(cache), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise SystemExit(f"prediction cache changed after prepare: {cache}")
        grouped[run] = grouped_indices(cache)
    stage = out / "builds_l2" / plan["plan_id"] / SOURCE_NAME
    roots = plan["root_ids"][shard::shards]
    layout = plan["attribute_layout"]
    reused = written = 0
    for rid in tqdm(roots, desc=f"L2 shard {shard}/{shards}", unit="cell"):
        dest = stage / str(rid)
        if dest.is_file() and dest.stat().st_size:
            reused += 1
            continue
        graph = torch.load(Path(plan["graph_cache"]) / f"{rid}.pt",
                           map_location="cpu", weights_only=False)
        vertices = graph.pos.numpy().astype(np.float32, copy=False)
        directed = graph.edge_index.numpy().T.astype(np.int64, copy=False)
        edges = directed[directed[:, 0] < directed[:, 1]].astype(np.uint32, copy=False)
        n = len(vertices)
        by_run = {}
        target_values = np.full(n, -1, np.float32)
        for run in runs:
            values = np.full(n, -1, np.float32)
            source = grouped[run].get(int(rid))
            if source is not None:
                centers, predictions, targets = source
                if len(np.unique(centers)) != len(centers):
                    raise ValueError(f"{run}/{rid}: duplicate center_index values")
                if centers.size and (centers.min() < 0 or centers.max() >= n):
                    raise ValueError(f"{run}/{rid}: center_index outside 0..{n - 1}")
                values[centers] = predictions
                unset = target_values[centers] < 0
                target_values[centers[unset]] = targets[unset]
            by_run[run] = values
        attrs = [VertexAttribute.from_values("target_class", target_values)]
        for start in range(0, len(runs), PREDICTIONS_PER_ATTRIBUTE):
            packed_runs = runs[start:start + PREDICTIONS_PER_ATTRIBUTE]
            packed_int = np.zeros((n, SCALARS_PER_ATTRIBUTE), np.uint32)
            for offset, run in enumerate(packed_runs):
                prediction = by_run[run].astype(np.int16, copy=False)
                if prediction.size and (prediction.min() < -1 or prediction.max() > 7):
                    raise ValueError(f"{run}/{rid}: prediction outside -1..7")
                component = offset // PREDICTIONS_PER_SCALAR
                nibble = offset % PREDICTIONS_PER_SCALAR
                packed_int[:, component] |= (
                    (prediction.astype(np.int32) + 1).astype(np.uint32)
                    << (BITS_PER_PREDICTION * nibble)
                )
            attr = layout[packed_runs[0]]["vertex_attribute"]
            attrs.append(VertexAttribute.from_values(attr, packed_int.astype(np.float32)))
        _atomic_write(dest, encode_skeleton(vertices, edges, attrs))
        written += 1
    print(f"L2 shard {shard}/{shards}: wrote {written}, reused {reused}")


def finalize(out: Path, plan_key: str | None) -> None:
    plan = load_plan(out, plan_key)
    build = out / "builds_l2" / plan["plan_id"]
    stage = build / SOURCE_NAME
    missing = [rid for rid in plan["root_ids"] if not (stage / str(rid)).is_file()]
    if missing:
        raise SystemExit(f"cannot finalize: {len(missing)} L2 payloads missing")
    props = build / "segment_properties_l2"
    if props.exists():
        shutil.rmtree(props)
    write_segment_properties(props, plan["root_ids"])
    info = json.loads((stage / "info").read_text())
    info["segment_properties"] = "../segment_properties_l2"
    _atomic_write(stage / "info", _json_bytes(info))
    labels = list(load_hierarchy(load_manifest()).level_classes[-1])
    metadata = {
        "description": "Exact-node predictions on embedding-covered CAVE L2 skeleton graphs.",
        "included_fold": plan["included_fold"], "manifest_split": "test",
        "fold_note": plan["fold_note"],
        "class_codes": {"-1": "unavailable", **{str(i): v for i, v in enumerate(labels)}},
        "models": [{"model_index": i, "run": run, **plan["attribute_layout"][run]}
                   for i, run in enumerate(plan["runs"])],
        "attribute_packing": {
            "encoding": "four-bit unsigned nibbles; stored value = class_code + 1",
            "unavailable_encoded": 0, "class_code_range": [0, 7],
            "predictions_per_float": PREDICTIONS_PER_SCALAR,
            "floats_per_attribute": SCALARS_PER_ATTRIBUTE,
            "predictions_per_attribute": PREDICTIONS_PER_ATTRIBUTE,
            "components": ["x", "y", "z", "w"],
        },
        "assignment": plan["join"], "unpredicted_nodes": "class code -1; no propagation",
        "skeleton_source": plan["graph_cache"], "source_name": SOURCE_NAME,
        "export_plan": plan["plan_id"],
    }
    _atomic_write(out / METADATA_NAME, _json_bytes(metadata))
    final, previous = out / SOURCE_NAME, out / f"{SOURCE_NAME}.previous"
    final_props, previous_props = out / "segment_properties_l2", out / "segment_properties_l2.previous"
    for old in (previous, previous_props):
        if old.exists():
            shutil.rmtree(old)
    if final.exists():
        os.replace(final, previous)
    if final_props.exists():
        os.replace(final_props, previous_props)
    os.replace(props, final_props)
    os.replace(stage, final)
    print(f"published {len(plan['root_ids'])} roots to {final}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--graph-cache", type=Path, default=GRAPH_CACHE)
    ap.add_argument("--phase", choices=("all", "prepare", "export", "finalize"), default="all")
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
    ap.add_argument("--plan-key", default=os.environ.get("PLAN_KEY"))
    args = ap.parse_args()
    if args.phase == "export":
        export(args.out, args.shard_index, args.num_shards, args.plan_key); return
    if args.phase == "finalize":
        finalize(args.out, args.plan_key); return
    runs = completed_runs(args.runs)
    key = args.plan_key or os.environ.get("SLURM_JOB_ID")
    prepare(runs, args.out, args.graph_cache, key)
    if args.phase == "all":
        export(args.out, 0, 1, key)
        finalize(args.out, key)


if __name__ == "__main__":
    main()
