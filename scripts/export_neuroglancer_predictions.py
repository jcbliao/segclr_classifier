"""Infer fixed-node models and export the held-out fold as one NG layer.

Inference results are cached per run.  The export is then rebuilt from those
caches, with one float32 vertex attribute per model.  Attribute values are
finest-level class codes; ``prediction_labels.json`` maps every code to the
specific class name.  A value of -1 means that root has no embedding graph.
Only the dataset's ``test`` fold is exported; validation aliases that fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.loader import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "all_windows"
DEFAULT_SKELS = Path("/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons")
DEFAULT_OUT = Path(
    "/orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/"
    "segclr_predictions/fold_test_validation"
)
DEFAULT_NB = Path("/orcd/scratch/orcd/013/jcbliao/embedding_paths/r5um")
DEFAULT_PUBLIC_URL = (
    "https://g-ffa18e.d1c26e.5898.data.globus.org/microns/"
    "segclr_predictions/fold_test_validation"
)
SKEL_REPO = Path("/home/jcbliao/rotation/skeletonization")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SKEL_REPO))

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402
from data.window_prediction_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR, load_prediction_cache, save_prediction_cache,
)
from gnn.model import WindowClassifier  # noqa: E402
from skeletonization.precomputed import (  # noqa: E402
    VertexAttribute, encode_skeleton,
)

PLAN_SCHEMA_VERSION = 4
BITS_PER_PREDICTION = 4
PREDICTIONS_PER_SCALAR = 6
SCALARS_PER_ATTRIBUTE = 4
PREDICTIONS_PER_ATTRIBUTE = PREDICTIONS_PER_SCALAR * SCALARS_PER_ATTRIBUTE


def completed_runs(requested: list[str]) -> list[str]:
    if requested:
        runs = requested
    else:
        # The checkpoint is the authoritative completed-model artifact.  A
        # top-level ``<run>.json`` is only a convenience summary and is absent
        # for valid runs produced or renamed by some batch workflows.
        runs = sorted(
            path.parent.name
            for path in RESULTS.glob("*_n*/checkpoint_best.pt")
        )
    missing = [
        r for r in runs
        if not (RESULTS / r / "checkpoint_best.pt").is_file()
    ]
    if missing:
        raise SystemExit("missing completed checkpoint: " + ", ".join(missing))
    return runs


def attribute_layout(runs: list[str]) -> dict[str, dict[str, int | str]]:
    """Map models into 4-bit lanes packed six-per-float, four floats per vec4."""
    return {
        run: {
            "vertex_attribute": f"prediction_pack_{i // PREDICTIONS_PER_ATTRIBUTE:03d}",
            "component": (i % PREDICTIONS_PER_ATTRIBUTE) // PREDICTIONS_PER_SCALAR,
            "nibble": i % PREDICTIONS_PER_SCALAR,
            "divisor": 16 ** (i % PREDICTIONS_PER_SCALAR),
        }
        for i, run in enumerate(runs)
    }


@torch.no_grad()
def infer_run(run: str, cache_dir: Path, batch_size: int, workers: int,
              neighborhood_root: Path, force: bool) -> Path:
    dest = cache_dir / f"{run}.npz"
    if dest.exists() and not force:
        try:
            load_prediction_cache(run, cache_dir, required_splits=("test",))
            print(f"cached: {dest}", flush=True)
            return dest
        except ValueError as exc:
            print(f"rebuilding {dest}: {exc}", flush=True)
    meta = json.loads((RESULTS / f"{run}.json").read_text())
    args = meta["args"]
    n = int(args["num_embeddings"])
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    checkpoint = torch.load(RESULTS / run / "checkpoint_best.pt",
                            map_location="cpu", weights_only=False)
    model = WindowClassifier(checkpoint["config"], hierarchy)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_split, all_root, all_center, all_xyz, all_pred, all_target = [], [], [], [], [], []
    # This project has no separate validation partition: validation is an alias
    # for test. Training-fold predictions do not belong in this visualization.
    for split in ("test",):
        ds = WindowedGraphDatasetLCPN(
            manifest, split, pos_dim=args["gt_pos_dim"],
            use_thickness=bool(args.get("gt_use_thickness", False)),
            num_embeddings=n, neighborhood_root=neighborhood_root,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, persistent_workers=workers > 0)
        pieces = []
        for batch in tqdm(loader, desc=f"{run} {split}"):
            batch = batch.to(device)
            hidden = model(batch.x, batch.edge_index, batch.batch,
                           batch.pos_enc, batch.rel_pos,
                           getattr(batch, "thickness", None))
            pieces.append(model.cls_head.predict_top_down(hidden)[:, -1].cpu().numpy())
        pred = np.concatenate(pieces).astype(np.int16)
        if len(pred) != len(ds):
            raise RuntimeError(f"{run}/{split}: {len(pred)} predictions for {len(ds)} windows")
        # Each fixed-node window is centered on this graph-cache node.
        xyz = np.empty((len(ds), 3), np.float32)
        for rid in np.unique(ds.index_root_ids):
            rows = np.flatnonzero(ds.index_root_ids == rid)
            centers = ds.index_centers[rows]
            xyz[rows] = ds.cell_data[int(rid)].pos[centers].numpy()
        all_root.append(ds.index_root_ids.astype(np.uint64))
        all_center.append(ds.index_centers.astype(np.int32))
        all_xyz.append(xyz)
        all_pred.append(pred)
        all_target.append(ds.index_labels.astype(np.int16))
        all_split.append(np.full(len(ds), split, dtype="U5"))

    save_prediction_cache(run, {
        "split": np.concatenate(all_split), "root_id": np.concatenate(all_root),
        "center_index": np.concatenate(all_center), "center_xyz": np.concatenate(all_xyz),
        "prediction": np.concatenate(all_pred), "target": np.concatenate(all_target),
        "num_embeddings": np.array([n], np.int16),
    }, cache_dir)
    print(f"wrote {dest}", flush=True)
    return dest


def grouped_cache(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    arrays = load_prediction_cache(path.stem, path.parent, required_splits=("test",))
    held_out = arrays["split"].astype(str) == "test"
    roots = arrays["root_id"][held_out].astype(np.uint64)
    xyz = arrays["center_xyz"][held_out].astype(np.float64)
    pred = arrays["prediction"][held_out].astype(np.float32)
    target = arrays["target"][held_out].astype(np.float32)
    order = np.argsort(roots, kind="stable")
    roots, xyz, pred, target = roots[order], xyz[order], pred[order], target[order]
    cuts = np.flatnonzero(np.diff(roots)) + 1
    return {int(r[0]): (x, p, t) for r, x, p, t in zip(
        np.split(roots, cuts), np.split(xyz, cuts), np.split(pred, cuts),
        np.split(target, cuts))}


def write_segment_properties(path: Path, roots: list[int]) -> None:
    payload = {
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": [str(r) for r in roots], "properties": [
            {"id": "label", "type": "label", "values": [str(r) for r in roots]},
            {"id": "fold", "type": "description",
             "values": ["test/validation"] * len(roots)},
        ]},
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "info").write_text(json.dumps(payload, indent=2))


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def prepare_export(runs: list[str], caches: list[Path], skel_dir: Path,
                   out: Path, plan_key: str | None = None) -> Path:
    """Freeze an immutable export plan shared by every array task."""
    manifest = load_manifest()
    held_out_roots = {
        int(root_id) for root_id, cell in manifest["cells"].items()
        if cell["split"] == "test"
    }
    roots = sorted(
        int(p.stem) for p in skel_dir.glob("*.npz")
        if p.stem.isdigit() and int(p.stem) in held_out_roots
    )
    if not roots:
        raise SystemExit(f"no skeleton npz files in {skel_dir}")
    layout = attribute_layout(runs)
    # Loading validates schema and test-fold completeness before array jobs are
    # submitted. Cache paths are immutable inputs for this plan in normal use.
    for run, cache in zip(runs, caches):
        load_prediction_cache(run, cache.parent, required_splits=("test",))
    cache_inputs = []
    for cache in caches:
        stat = cache.stat()
        cache_inputs.append({
            "path": str(cache), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        })
    plan_body = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "runs": runs,
        "cache_inputs": cache_inputs,
        "attribute_layout": layout,
        "skeleton_source": str(skel_dir),
        "root_ids": roots,
        "included_fold": "test/validation",
        "manifest_split": "test",
        "fold_note": (
            "Held-out whole-cell test fold. Validation aliases this exact fold; "
            "no training-fold cells are included."
        ),
    }
    plan_id = hashlib.sha256(
        json.dumps(plan_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    plan = {**plan_body, "plan_id": plan_id}
    build = out / "builds" / plan_id
    skeletons = build / "skeletons"
    skeletons.mkdir(parents=True, exist_ok=True)
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "vertex_attributes": [
            {"id": "radius", "data_type": "float32", "num_components": 1},
            {"id": "target_class", "data_type": "float32", "num_components": 1},
            *[
                {
                    "id": f"prediction_pack_{pack:03d}",
                    "data_type": "float32",
                    "num_components": SCALARS_PER_ATTRIBUTE,
                }
                for pack in range(
                    (len(runs) + PREDICTIONS_PER_ATTRIBUTE - 1)
                    // PREDICTIONS_PER_ATTRIBUTE
                )
            ],
        ],
    }
    _atomic_write(skeletons / "info", _json_bytes(info))
    _atomic_write(build / "plan.json", _json_bytes(plan))
    _atomic_write(out / "active_plan.json", _json_bytes(plan))
    if plan_key:
        _atomic_write(out / "plans" / f"{plan_key}.json", _json_bytes(plan))
    print(f"prepared export plan {plan_id}: {len(roots)} roots, {len(runs)} models")
    print(f"array build directory: {build}")
    return out / "active_plan.json"


def load_plan(out: Path, plan_key: str | None = None) -> dict:
    path = (out / "plans" / f"{plan_key}.json" if plan_key
            else out / "active_plan.json")
    if not path.is_file():
        raise SystemExit(f"missing export plan: {path}; run --phase prepare first")
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise SystemExit(f"unsupported export plan schema in {path}")
    frozen = out / "builds" / plan["plan_id"] / "plan.json"
    if not frozen.is_file() or json.loads(frozen.read_text()) != plan:
        raise SystemExit(f"active plan does not match frozen plan {frozen}")
    return plan


def export_shard(out: Path, shard_index: int, num_shards: int,
                 plan_key: str | None = None) -> None:
    """Atomically write one disjoint, independently resumable root shard."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise SystemExit(f"invalid shard {shard_index}/{num_shards}")
    plan = load_plan(out, plan_key)
    runs = list(plan["runs"])
    caches = []
    for expected in plan["cache_inputs"]:
        cache = Path(expected["path"])
        stat = cache.stat()
        actual = {"path": str(cache), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if actual != expected:
            raise SystemExit(
                f"prediction cache changed after plan preparation: {cache}; "
                "run --phase prepare again"
            )
        caches.append(cache)
    grouped = {run: grouped_cache(cache) for run, cache in zip(runs, caches)}
    skel_dir = Path(plan["skeleton_source"])
    dest_dir = out / "builds" / plan["plan_id"] / "skeletons"
    roots = [int(r) for r in plan["root_ids"]][shard_index::num_shards]
    layout = plan["attribute_layout"]
    written = skipped = 0
    for rid in tqdm(roots, desc=f"export shard {shard_index}/{num_shards}", unit="cell"):
        dest = dest_dir / str(rid)
        if dest.is_file() and dest.stat().st_size > 0:
            skipped += 1
            continue
        with np.load(skel_dir / f"{rid}.npz") as z:
            vertices = z["vertices"].astype(np.float32)
            edges = z["edges"].astype(np.uint32)
            radius = z["radius"].astype(np.float32)
        attrs = [VertexAttribute.from_values("radius", radius)]
        predictions_by_run = {}
        target_values = None
        for run in runs:
            source = grouped[run].get(rid)
            if source is None:
                values = np.full(len(vertices), -1, np.float32)
            else:
                center_xyz, predictions, targets = source
                nearest = cKDTree(center_xyz).query(vertices, k=1)[1]
                values = predictions[nearest].astype(np.float32, copy=False)
                if target_values is None:
                    target_values = targets[nearest].astype(np.float32, copy=False)
            predictions_by_run[run] = values
        if target_values is None:
            target_values = np.full(len(vertices), -1, np.float32)
        if target_values.size and (target_values.min() < -1 or target_values.max() > 7):
            raise ValueError(f"{rid}: target outside supported class codes -1..7")
        attrs.append(VertexAttribute.from_values("target_class", target_values))
        for start in range(0, len(runs), PREDICTIONS_PER_ATTRIBUTE):
            packed_runs = runs[start:start + PREDICTIONS_PER_ATTRIBUTE]
            # Encode unavailable=-1 as 0 and classes 0..7 as 1..8. Six 4-bit
            # nibbles occupy at most 24 bits, which float32 represents exactly.
            packed_int = np.zeros((len(vertices), SCALARS_PER_ATTRIBUTE), np.uint32)
            for offset, run in enumerate(packed_runs):
                prediction = predictions_by_run[run].astype(np.int16, copy=False)
                if prediction.size and (prediction.min() < -1 or prediction.max() > 7):
                    raise ValueError(
                        f"{run}/{rid}: prediction outside supported class codes -1..7"
                    )
                component = offset // PREDICTIONS_PER_SCALAR
                nibble = offset % PREDICTIONS_PER_SCALAR
                encoded = (prediction.astype(np.int32) + 1).astype(np.uint32)
                packed_int[:, component] |= encoded << (BITS_PER_PREDICTION * nibble)
            packed = packed_int.astype(np.float32)
            attribute = layout[packed_runs[0]]["vertex_attribute"]
            attrs.append(VertexAttribute.from_values(attribute, packed))
        payload = encode_skeleton(vertices, edges, attrs)
        _atomic_write(dest, payload)
        written += 1
    print(f"shard {shard_index}/{num_shards}: wrote {written}, reused {skipped}, total {len(roots)}")


def finalize_export(out: Path, plan_key: str | None = None) -> None:
    """Publish a complete build atomically; refuse incomplete array output."""
    plan = load_plan(out, plan_key)
    runs = list(plan["runs"])
    roots = [int(r) for r in plan["root_ids"]]
    layout = plan["attribute_layout"]
    build = out / "builds" / plan["plan_id"]
    stage = build / "skeletons"
    published_manifest = out / "prediction_labels.json"
    if not stage.is_dir() and published_manifest.is_file():
        published = json.loads(published_manifest.read_text())
        if published.get("export_plan") == plan["plan_id"]:
            print(f"export plan {plan['plan_id']} is already published")
            return
    missing = [rid for rid in roots if not (stage / str(rid)).is_file()
               or (stage / str(rid)).stat().st_size == 0]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise SystemExit(
            f"cannot finalize plan {plan['plan_id']}: {len(missing)}/{len(roots)} "
            f"root payloads missing or empty ({preview})"
        )

    out.mkdir(parents=True, exist_ok=True)
    final = out / "skeletons"
    backup = out / "skeletons.previous"
    staged_properties = build / "segment_properties"
    if staged_properties.exists():
        shutil.rmtree(staged_properties)
    write_segment_properties(staged_properties, roots)
    info = json.loads((stage / "info").read_text())
    info["segment_properties"] = "../segment_properties"
    _atomic_write(stage / "info", _json_bytes(info))

    hierarchy = load_hierarchy(load_manifest())
    labels = list(hierarchy.level_classes[-1])
    manifest = {
        "description": "Per-node finest-level predictions. -1 means unavailable.",
        "included_fold": plan["included_fold"],
        "manifest_split": plan["manifest_split"],
        "fold_note": plan["fold_note"],
        "class_codes": {"-1": "unavailable", **{str(i): v for i, v in enumerate(labels)}},
        "models": [{"run": run, **layout[run]} for run in runs],
        "attribute_packing": {
            "encoding": "four-bit unsigned nibbles; stored value = class_code + 1",
            "unavailable_encoded": 0,
            "class_code_range": [0, 7],
            "predictions_per_float": PREDICTIONS_PER_SCALAR,
            "floats_per_attribute": SCALARS_PER_ATTRIBUTE,
            "predictions_per_attribute": PREDICTIONS_PER_ATTRIBUTE,
            "components": ["x", "y", "z", "w"],
            "glsl_decode": (
                "floor(mod(floor(component_value / divisor), 16.0)) - 1.0"
            ),
        },
        "propagation": "nearest fixed-window center within the same root ID",
        "skeleton_source": plan["skeleton_source"],
        "export_plan": plan["plan_id"],
    }
    _atomic_write(out / "prediction_labels.json", _json_bytes(manifest))
    state = {
        "dimensions": {d: [1e-9, "m"] for d in "xyz"},
        "layers": [{"type": "segmentation", "name": "segclr_predictions_test_validation",
                    "source": f"precomputed://{DEFAULT_PUBLIC_URL}/skeletons",
                    "segments": [str(roots[0])]}], "layout": "3d",
    }
    _atomic_write(out / "viewer_state.template.json", _json_bytes(state))
    properties = out / "segment_properties"
    properties_backup = out / "segment_properties.previous"
    if properties_backup.exists():
        shutil.rmtree(properties_backup)
    if properties.exists():
        os.replace(properties, properties_backup)
    os.replace(staged_properties, properties)
    # The visible skeleton source changes last, after all metadata is ready.
    if backup.exists():
        shutil.rmtree(backup)
    if final.exists():
        os.replace(final, backup)
    os.replace(stage, final)
    print(f"exported {len(roots)} selectable root IDs and {len(runs)} models to {out}")
    if backup.exists():
        print(f"previous skeleton source retained at {backup}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", help="completed run names; default: all completed fixed-node runs")
    ap.add_argument("--skeleton-dir", type=Path, default=DEFAULT_SKELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--neighborhood-root", type=Path, default=DEFAULT_NB)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=15)
    ap.add_argument("--force-inference", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--phase", choices=("all", "prepare", "export", "finalize"),
                    default="all")
    ap.add_argument("--shard-index", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ap.add_argument("--num-shards", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
    ap.add_argument("--plan-key", default=os.environ.get("PLAN_KEY"),
                    help="isolated plan pointer (the submitter uses the prepare job ID)")
    args = ap.parse_args()
    if args.phase == "export":
        export_shard(args.out, args.shard_index, args.num_shards, args.plan_key)
        return
    if args.phase == "finalize":
        finalize_export(args.out, args.plan_key)
        return
    runs = completed_runs(args.runs)
    if not runs:
        raise SystemExit("no completed fixed-node runs found")
    cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    caches = []
    for run in runs:
        cache = cache_dir / f"{run}.npz"
        if args.export_only:
            if not cache.exists():
                raise SystemExit(f"missing cache for --export-only: {cache}")
        else:
            infer_run(run, cache_dir, args.batch_size, args.num_workers,
                      args.neighborhood_root, args.force_inference)
        caches.append(cache)
    plan_key = args.plan_key or os.environ.get("SLURM_JOB_ID")
    prepare_export(runs, caches, args.skeleton_dir, args.out, plan_key)
    if args.phase == "all":
        export_shard(args.out, 0, 1, plan_key)
        finalize_export(args.out, plan_key)


if __name__ == "__main__":
    main()
