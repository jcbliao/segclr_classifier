"""Build per-window prediction/geometry caches for feature_prediction_correlation.ipynb."""
from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from data.dataset_lcpn import load_hierarchy, load_manifest
from data.dataset_windowed import WindowedGraphDatasetLCPN
from data.window_prediction_cache import (
    DEFAULT_CACHE_DIR, load_prediction_cache, save_prediction_cache,
)
from gnn.model import WindowClassifier

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = ROOT / "analysis" / "feature_prediction_cache"
NB_ROOT = Path("/orcd/scratch/orcd/013/jcbliao/embedding_paths/r5um")
NEW_SKEL = Path("/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons")
VOLUME = ROOT / "data" / "mask_volume_cache"
NUCLEI = ROOT / "data" / "nucleus_positions.json"
NM3_PER_UM3 = 1e9


def completed_runs() -> list[str]:
    return sorted(p.stem for p in RESULTS.glob("*_n*.json") if (RESULTS / p.stem / "checkpoint_best.pt").exists())


def _reduce(values, offsets, op):
    out = np.full(len(offsets) - 1, np.nan, np.float32)
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        x = values[lo:hi]
        x = x[np.isfinite(x)]
        if len(x):
            out[i] = op(x)
    return out


def _soma_distances(data, nucleus_xyz):
    pos = data.pos.numpy().astype(np.float64)
    spatial = np.linalg.norm(pos - nucleus_xyz, axis=1).astype(np.float32)
    anchor = int(np.argmin(spatial))
    ei = data.edge_index.numpy()
    ew = data.edge_attr.numpy().reshape(-1)
    graph = coo_matrix((ew, (ei[0], ei[1])), shape=(len(pos), len(pos))).tocsr()
    path = dijkstra(graph, directed=False, indices=anchor).astype(np.float32)
    return spatial, path


def _new_skeleton_radius(root_id, positions):
    path = NEW_SKEL / f"{root_id}.npz"
    if not path.exists():
        return np.full(len(positions), np.nan, np.float32)
    with np.load(path) as z:
        vertices = z["vertices"].astype(np.float64)
        radius = z["radius"].astype(np.float32)
    _, nearest = cKDTree(vertices).query(positions, k=1)
    return radius[nearest]


def _volume(root_id, data):
    path = VOLUME / f"{root_id}.npz"
    if not path.exists():
        return np.full(data.num_nodes, np.nan, np.float32)
    with np.load(path) as z:
        if not np.array_equal(z["orig_node_ids"], data.orig_node_ids.numpy()):
            raise RuntimeError(f"volume/node-id mismatch for {root_id}")
        scale = int(z["voxel_volume_nm3"]) / NM3_PER_UM3
        return (z["voxel_count"] * scale).astype(np.float32)


@torch.no_grad()
def build_cache(run_name: str, batch_size=1024, num_workers=15, force=False) -> Path:
    """Run held-out inference and join per-window physical features once."""
    CACHE.mkdir(exist_ok=True)
    dest = CACHE / f"{run_name}.npz"
    summary = CACHE / f"{run_name}.summary.json"
    if dest.exists() and not force:
        if not summary.exists():
            summarize_cache(run_name)
        return dest
    meta = json.loads((RESULTS / f"{run_name}.json").read_text())
    n = int(meta["args"]["num_embeddings"])
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    ds = WindowedGraphDatasetLCPN(manifest, "test", pos_dim=meta["args"]["gt_pos_dim"],
                                  num_embeddings=n, neighborhood_root=NB_ROOT)
    try:
        shared = load_prediction_cache(run_name, DEFAULT_CACHE_DIR, required_splits=("test",))
        keep = shared["split"].astype(str) == "test"
        if not np.array_equal(shared["root_id"][keep].astype(np.int64), ds.index_root_ids):
            raise ValueError("test root ordering differs from current dataset")
        if not np.array_equal(shared["center_index"][keep].astype(np.int64), ds.index_centers):
            raise ValueError("test center ordering differs from current dataset")
        correct = shared["prediction"][keep].astype(np.int64) == ds.index_labels
        print(f"using shared predictions from {DEFAULT_CACHE_DIR}", flush=True)
    except (FileNotFoundError, ValueError):
        checkpoint = torch.load(RESULTS / run_name / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        model = WindowClassifier(checkpoint["config"], hierarchy)
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            persistent_workers=num_workers > 0)
        predictions = []
        for batch in tqdm(loader, desc=f"infer {run_name}"):
            batch = batch.to(device)
            g = model(batch.x, batch.edge_index, batch.batch, batch.pos_enc, batch.rel_pos,
                      getattr(batch, "thickness", None))
            pred = model.cls_head.predict_top_down(g)[:, -1]
            predictions.append(pred.cpu().numpy())
        predictions = np.concatenate(predictions).astype(np.int16)
        correct = predictions == ds.index_labels
        xyz = np.empty((len(ds), 3), np.float32)
        for rid in np.unique(ds.index_root_ids):
            rows = np.flatnonzero(ds.index_root_ids == rid)
            xyz[rows] = ds.cell_data[int(rid)].pos[ds.index_centers[rows]].numpy()
        save_prediction_cache(run_name, {
            "split": np.full(len(ds), "test", dtype="U5"),
            "root_id": ds.index_root_ids.astype(np.uint64),
            "center_index": ds.index_centers.astype(np.int32),
            "center_xyz": xyz, "prediction": predictions,
            "target": ds.index_labels.astype(np.int16),
            "num_embeddings": np.array([n], np.int16),
        }, DEFAULT_CACHE_DIR)

    nuclei = {int(k): np.asarray(v, np.float64) for k, v in json.loads(NUCLEI.read_text())["positions"].items()}
    columns = {k: [] for k in (
        "root_id", "cell_type", "geodesic_radius_um", "path_distance_um",
        "node_density_per_um", "spatial_radius_um", "spatial_density_per_um3",
        "soma_path_um", "soma_spatial_um", "volume_sum_um3", "volume_mean_um3",
        "volume_median_um3", "radius_mean_nm", "radius_median_nm", "radius_min_nm",
        "radius_max_nm")}
    for root_id, data in tqdm(ds.cell_data.items(), desc=f"features n={n}", unit="cell"):
        with np.load(NB_ROOT / "neighborhoods" / f"n{n}" / f"{root_id}.npz") as z:
            offsets = z["offsets"].astype(np.int64)
            restricted_members = z["members"].astype(np.int64)
            cache_index = z["cache_index"].astype(np.int64)
            cable = z["cable_nm"].astype(np.float32)
            geodesic_radius = z["radius_nm"].astype(np.float32)
        members = cache_index[restricted_members]
        centers = cache_index
        pos = data.pos.numpy().astype(np.float64)
        repeated_centers = np.repeat(pos[centers], np.diff(offsets), axis=0)
        euclid = np.linalg.norm(pos[members] - repeated_centers, axis=1)
        spatial_radius = _reduce(euclid, offsets, np.max)
        volumes = _volume(root_id, data)[members]
        new_radius = _new_skeleton_radius(root_id, pos)[members]
        if root_id in nuclei:
            soma_spatial, soma_path = _soma_distances(data, nuclei[root_id])
            soma_spatial, soma_path = soma_spatial[centers], soma_path[centers]
        else:
            soma_spatial = soma_path = np.full(len(centers), np.nan, np.float32)
        cell_type = manifest["cells"][str(root_id)]["cell_type"]
        columns["root_id"].append(np.full(len(centers), root_id, np.int64))
        columns["cell_type"].append(np.full(len(centers), cell_type, dtype=f"U{max(16, len(cell_type))}"))
        columns["geodesic_radius_um"].append(geodesic_radius / 1000)
        columns["path_distance_um"].append(cable / 1000)
        columns["node_density_per_um"].append(n / np.maximum(cable / 1000, 1e-6))
        columns["spatial_radius_um"].append(spatial_radius / 1000)
        columns["spatial_density_per_um3"].append(n / np.maximum(4/3*np.pi*(spatial_radius/1000)**3, 1e-9))
        columns["soma_path_um"].append(soma_path / 1000)
        columns["soma_spatial_um"].append(soma_spatial / 1000)
        columns["volume_sum_um3"].append(_reduce(volumes, offsets, np.sum))
        columns["volume_mean_um3"].append(_reduce(volumes, offsets, np.mean))
        columns["volume_median_um3"].append(_reduce(volumes, offsets, np.median))
        columns["radius_mean_nm"].append(_reduce(new_radius, offsets, np.mean))
        columns["radius_median_nm"].append(_reduce(new_radius, offsets, np.median))
        columns["radius_min_nm"].append(_reduce(new_radius, offsets, np.min))
        columns["radius_max_nm"].append(_reduce(new_radius, offsets, np.max))
    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    if len(correct) != len(arrays["root_id"]):
        raise RuntimeError(f"prediction/feature row mismatch: {len(correct)} != {len(arrays['root_id'])}")
    np.savez_compressed(dest, correct=correct, n_embeddings=np.array([n]), **arrays)
    summarize_cache(run_name)
    return dest


def load_cache(run_name: str):
    import pandas as pd
    with np.load(CACHE / f"{run_name}.npz") as z:
        return pd.DataFrame({k: z[k] for k in z.files if k != "n_embeddings"})


FEATURES = [
    "geodesic_radius_um", "path_distance_um", "node_density_per_um",
    "spatial_radius_um", "spatial_density_per_um3", "soma_path_um",
    "soma_spatial_um", "volume_sum_um3", "volume_mean_um3",
    "volume_median_um3", "radius_mean_nm", "radius_median_nm",
    "radius_min_nm", "radius_max_nm",
]


def summarize_cache(run_name: str, bins=10) -> Path:
    """Reduce the multi-million-row cache to login-node-safe plot tables."""
    import pandas as pd
    frame = load_cache(run_name)
    n_embeddings = int(re.search(r"_n(10|20|40)$", run_name).group(1))
    correlations, binned = [], []

    def add_group(part, cell_type=None):
        for feature in FEATURES:
            if cell_type is not None and feature not in ("soma_path_um", "soma_spatial_um"):
                continue
            x = part[[feature, "correct"]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p = spearmanr(x[feature], x.correct) if len(x) > 2 else (np.nan, np.nan)
            correlations.append({"run": run_name, "n_embeddings": n_embeddings,
                                 "cell_type": cell_type, "feature": feature,
                                 "spearman_rho": float(rho), "p": float(p), "n": len(x)})
            if len(x) < bins or x[feature].nunique() < 2:
                continue
            x = x.assign(bin=pd.qcut(x[feature], bins, duplicates="drop"))
            table = x.groupby("bin", observed=True).agg(
                feature_median=(feature, "median"), accuracy=("correct", "mean"),
                n=("correct", "size")).reset_index(drop=True)
            for rank, row in table.iterrows():
                binned.append({"run": run_name, "n_embeddings": n_embeddings,
                               "cell_type": cell_type, "feature": feature, "bin": int(rank),
                               "feature_median": float(row.feature_median),
                               "accuracy": float(row.accuracy), "n": int(row.n)})

    add_group(frame)
    for cell_type, part in frame.groupby("cell_type"):
        add_group(part, str(cell_type))
    cells = (frame.groupby(["root_id", "cell_type"], as_index=False)
             .agg(accuracy=("correct", "mean"), n_windows=("correct", "size")))
    cell_rows = [{"run": run_name, "n_embeddings": n_embeddings, **row}
                 for row in cells.to_dict(orient="records")]
    payload = {"run": run_name, "n_embeddings": n_embeddings,
               "correlations": correlations, "bins": binned, "cells": cell_rows}
    out = CACHE / f"{run_name}.summary.json"
    out.write_text(json.dumps(payload))
    return out


def load_summary(run_name: str) -> dict:
    return json.loads((CACHE / f"{run_name}.summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_names", nargs="*", help="completed result run names; default: all")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runs = args.run_names or completed_runs()
    if not runs:
        raise SystemExit("no completed fixed-node runs found")
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run}", flush=True)
        print(build_cache(run, batch_size=args.batch_size, num_workers=args.num_workers,
                          force=args.force), flush=True)


if __name__ == "__main__":
    main()
