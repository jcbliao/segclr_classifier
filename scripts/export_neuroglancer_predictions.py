"""Infer fixed-node models and export all new skeletons as one NG layer.

Inference results are cached per run.  The export is then rebuilt from those
caches, with one float32 vertex attribute per model.  Attribute values are
finest-level class codes; ``prediction_labels.json`` maps every code to the
specific class name.  A value of -1 means that root has no embedding graph.
"""
from __future__ import annotations

import argparse
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
RESULTS = REPO / "results"
DEFAULT_SKELS = Path("/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons")
DEFAULT_OUT = Path("/orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/segclr_predictions")
DEFAULT_NB = Path("/orcd/scratch/orcd/013/jcbliao/embedding_paths/r5um")
SKEL_REPO = Path("/home/jcbliao/rotation/skeletonization")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SKEL_REPO))

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402
from data.window_prediction_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR, load_prediction_cache, save_prediction_cache,
)
from gnn.model import WindowClassifier  # noqa: E402
from skeletonization.precomputed import write_skeleton  # noqa: E402


def completed_runs(requested: list[str]) -> list[str]:
    if requested:
        runs = requested
    else:
        runs = sorted(p.stem for p in RESULTS.glob("*_n*.json"))
    missing = [r for r in runs if not (RESULTS / f"{r}.json").is_file()
               or not (RESULTS / r / "checkpoint_best.pt").is_file()]
    if missing:
        raise SystemExit("missing completed result/checkpoint: " + ", ".join(missing))
    return runs


def attribute_ids(runs: list[str]) -> dict[str, str]:
    # Short stable ids keep the skeleton info and shader expressions manageable.
    return {run: f"prediction_{i:03d}" for i, run in enumerate(runs)}


@torch.no_grad()
def infer_run(run: str, cache_dir: Path, batch_size: int, workers: int,
              neighborhood_root: Path, force: bool) -> Path:
    dest = cache_dir / f"{run}.npz"
    if dest.exists() and not force:
        try:
            load_prediction_cache(run, cache_dir, required_splits=("train", "test"))
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
    for split in ("train", "test"):
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


def grouped_cache(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    arrays = load_prediction_cache(path.stem, path.parent)
    roots = arrays["root_id"].astype(np.uint64)
    xyz = arrays["center_xyz"].astype(np.float64)
    pred = arrays["prediction"].astype(np.float32)
    order = np.argsort(roots, kind="stable")
    roots, xyz, pred = roots[order], xyz[order], pred[order]
    cuts = np.flatnonzero(np.diff(roots)) + 1
    return {int(r[0]): (x, p) for r, x, p in zip(
        np.split(roots, cuts), np.split(xyz, cuts), np.split(pred, cuts))}


def write_segment_properties(path: Path, roots: list[int]) -> None:
    payload = {
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": [str(r) for r in roots], "properties": [
            {"id": "label", "type": "label", "values": [str(r) for r in roots]}
        ]},
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "info").write_text(json.dumps(payload, indent=2))


def export_layer(runs: list[str], caches: list[Path], skel_dir: Path,
                 out: Path, force: bool) -> None:
    roots = sorted(int(p.stem) for p in skel_dir.glob("*.npz") if p.stem.isdigit())
    if not roots:
        raise SystemExit(f"no skeleton npz files in {skel_dir}")
    ids = attribute_ids(runs)
    grouped = {run: grouped_cache(cache) for run, cache in zip(runs, caches)}
    stage = out / f".skeletons.build.{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for rid in tqdm(roots, desc="write skeletons", unit="cell"):
        with np.load(skel_dir / f"{rid}.npz") as z:
            vertices = z["vertices"].astype(np.float32)
            edges = z["edges"].astype(np.uint32)
            radius = z["radius"].astype(np.float32)
        attrs = {}
        for run in runs:
            source = grouped[run].get(rid)
            if source is None:
                values = np.full(len(vertices), -1, np.float32)
            else:
                center_xyz, predictions = source
                nearest = cKDTree(center_xyz).query(vertices, k=1)[1]
                values = predictions[nearest].astype(np.float32, copy=False)
            attrs[ids[run]] = values
        write_skeleton(stage, rid, vertices, edges, radius=radius,
                       extra_attributes=attrs)

    out.mkdir(parents=True, exist_ok=True)
    final = out / "skeletons"
    backup = out / "skeletons.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if final.exists():
        os.replace(final, backup)
    os.replace(stage, final)
    write_segment_properties(out / "segment_properties", roots)
    info_path = final / "info"
    info = json.loads(info_path.read_text())
    info["segment_properties"] = "../segment_properties"
    info_path.write_text(json.dumps(info, indent=2))

    hierarchy = load_hierarchy(load_manifest())
    labels = list(hierarchy.level_classes[-1])
    manifest = {
        "description": "Per-node finest-level predictions. -1 means unavailable.",
        "class_codes": {"-1": "unavailable", **{str(i): v for i, v in enumerate(labels)}},
        "models": [{"run": run, "vertex_attribute": ids[run]} for run in runs],
        "propagation": "nearest fixed-window center within the same root ID",
        "skeleton_source": str(skel_dir),
    }
    (out / "prediction_labels.json").write_text(json.dumps(manifest, indent=2))
    state = {
        "dimensions": {d: [1e-9, "m"] for d in "xyz"},
        "layers": [{"type": "segmentation", "name": "segclr_predictions",
                    "source": "precomputed://REPLACE_WITH_HTTP_URL/skeletons",
                    "segments": []}], "layout": "3d",
    }
    (out / "viewer_state.template.json").write_text(json.dumps(state, indent=2))
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
    args = ap.parse_args()
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
    export_layer(runs, caches, args.skeleton_dir, args.out, force=True)


if __name__ == "__main__":
    main()
