"""Build matched CAVE/new-skeleton presynaptic-axon window databases.

The output is deliberately separate from ``data/graph_cache``.  For every
v1718 cell whose ``proofreading_status_and_strategy.status_axon`` is true it:

* locates the CAVE skeleton root as the CAVE node nearest the cached soma point;
* removes nodes within 5 um of that root coordinate from both skeletons;
* maps each surviving SegCLR-observed CAVE node to its nearest surviving new
  skeleton node, accepting only matches <= 5 um and averaging collisions;
* locates each presynaptic point on an observed node (again <= 5 um);
* grows a nearest-first geodesic window until K=10 observed nodes have been
  reached, retaining every intermediate, unobserved new-skeleton node; and
* stores position, embedding mask/features, induced edges, and window-local LPE.

The build is resumable and array-safe.  Run ``--prepare`` once (the only CAVE
query), array-build the cells, then run ``--finalize`` to create the manifest
and three cutoff figures.  See ``scripts/sbatch/build_presynaptic_axons.sh``.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "segclr_db" / "src"))

from data.geodesic_window import (  # noqa: E402
    DEFAULT_POS_DIM,
    _window_laplacian_pos_enc,
    build_csr_from_edges,
)

MAT_VERSION = 1718
DATASTACK = "minnie65_public"
PROOFREADING_TABLE = "proofreading_status_and_strategy"
SOMA_RADIUS_NM = 5_000.0
EMBEDDING_MAP_CUTOFF_NM = 2_000.0
SYNAPSE_MATCH_CUTOFF_NM = 5_000.0
DEFAULT_K = 10
DEFAULT_OUT = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10")
NEW_SKELETONS = Path("/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons")
PRESYNAPTIC = ROOT / "data" / "synapse_cache" / "presynaptic_sites.parquet"
NUCLEI = ROOT / "data" / "nucleus_positions.json"
MANIFEST = ROOT / "data" / "manifest.json"
GRAPH_CACHE = ROOT / "data" / "graph_cache"
CAVE_CACHE = ROOT / "data" / "skeleton_cache"


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def prepare(out: Path, token: str) -> None:
    """Cache the exact v1718 status_axon=True cohort; no other mode hits CAVE."""
    if not token:
        raise SystemExit("CAVE_TOKEN is required for --prepare")
    import caveclient

    client = caveclient.CAVEclient(DATASTACK, auth_token=token, version=MAT_VERSION)
    frame = client.materialize.query_table(
        PROOFREADING_TABLE, materialization_version=MAT_VERSION
    )
    from pandas.api.types import is_bool_dtype
    if "status_axon" not in frame or not is_bool_dtype(frame["status_axon"].dtype):
        raise TypeError(
            f"expected boolean status_axon; columns/dtypes are {frame.dtypes.to_dict()}"
        )
    root_col = "pt_root_id" if "pt_root_id" in frame else "root_id"
    axon_ok = frame["status_axon"].fillna(False).astype(bool)
    ids = sorted({int(x) for x in frame.loc[axon_ok, root_col].dropna()})
    payload = {
        "datastack": DATASTACK,
        "materialization_version": MAT_VERSION,
        "table": PROOFREADING_TABLE,
        "predicate": "status_axon == True (completed or partially completed axon)",
        "table_rows": len(frame),
        "root_ids": ids,
    }
    _atomic_json(out / "proofread_axon_root_ids.json", payload)
    print(f"cached {len(ids):,} proofread-axon root IDs in {out}", flush=True)


def _induced_cut(pos: np.ndarray, edges: np.ndarray, root_xyz: np.ndarray):
    distance = np.linalg.norm(pos.astype(np.float64) - root_xyz[None, :], axis=1)
    keep = distance > SOMA_RADIUS_NM
    remap = np.full(len(pos), -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    edge_keep = keep[edges[:, 0]] & keep[edges[:, 1]] if len(edges) else np.zeros(0, bool)
    cut_edges = remap[edges[edge_keep]].astype(np.int32)
    cut_pos = pos[keep].astype(np.float32)
    weights = (
        np.linalg.norm(
            cut_pos[cut_edges[:, 0]].astype(np.float64)
            - cut_pos[cut_edges[:, 1]].astype(np.float64), axis=1
        ).astype(np.float32)
        if len(cut_edges) else np.zeros(0, np.float32)
    )
    return keep, remap, cut_pos, cut_edges, weights, distance.astype(np.float32)


def _sym_edges(edges: np.ndarray) -> np.ndarray:
    if not len(edges):
        return np.zeros((2, 0), np.int64)
    return np.concatenate([edges.T, edges[:, ::-1].T], axis=1).astype(np.int64)


def _window(
    center: int,
    offsets: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    observed: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float] | None:
    """Dijkstra ball ending at the kth observed node; includes intermediates."""
    heap = [(0.0, int(center))]
    best = {int(center): 0.0}
    settled: list[int] = []
    seen: set[int] = set()
    n_observed = 0
    radius = 0.0
    while heap:
        dist, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        settled.append(node)
        radius = dist
        n_observed += int(observed[node])
        if n_observed == k:
            return np.asarray(settled, np.int32), float(radius)
        for p in range(int(offsets[node]), int(offsets[node + 1])):
            nxt = int(neighbors[p])
            nd = dist + float(weights[p])
            if nd < best.get(nxt, np.inf):
                best[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return None


def _window_arrays(
    points: pd.DataFrame,
    pos: np.ndarray,
    edges: np.ndarray,
    edge_length: np.ndarray,
    observed: np.ndarray,
    k: int,
    match_cutoff_nm: float = SYNAPSE_MATCH_CUTOFF_NM,
) -> dict[str, np.ndarray]:
    if not np.any(observed):
        n = len(points)
        return {
            "synapse_id": points["synapse_id"].to_numpy(np.int64),
            "synapse_xyz_nm": points[["cell_x_nm", "cell_y_nm", "cell_z_nm"]].to_numpy(np.float32),
            "nearest_observed_node": np.full(n, -1, np.int32),
            "nearest_observed_distance_nm": np.full(n, np.inf, np.float32),
            "within_match_cutoff": np.zeros(n, bool), "valid_k_window": np.zeros(n, bool),
            "radius_nm": np.full(n, np.nan, np.float32),
            "n_intermediate": np.zeros(n, np.int32),
            "window_offsets": np.zeros(n + 1, np.int64),
            "window_members": np.zeros(0, np.int32),
            "window_lpe": np.zeros((0, DEFAULT_POS_DIM), np.float32),
        }
    tree = cKDTree(pos[observed])
    observed_ids = np.flatnonzero(observed)
    xyz = points[["cell_x_nm", "cell_y_nm", "cell_z_nm"]].to_numpy(np.float64)
    nearest_dist, nearest_local = tree.query(xyz, k=1)
    centers = observed_ids[nearest_local].astype(np.int32)
    cutoff_ok = nearest_dist <= match_cutoff_nm

    offsets, neighbors, weights = build_csr_from_edges(
        _sym_edges(edges), np.concatenate([edge_length, edge_length])[:, None], len(pos)
    )
    member_offsets = [0]
    members: list[np.ndarray] = []
    lpe: list[np.ndarray] = []
    radii = np.full(len(points), np.nan, np.float32)
    valid = np.zeros(len(points), bool)
    n_intermediate = np.zeros(len(points), np.int32)
    for i in np.flatnonzero(cutoff_ok):
        result = _window(int(centers[i]), offsets, neighbors, weights, observed, k)
        if result is None:
            member_offsets.append(member_offsets[-1])
            continue
        nodes, radius = result
        local_map = np.full(len(pos), -1, np.int64)
        local_map[nodes] = np.arange(len(nodes))
        edge_keep = np.isin(edges[:, 0], nodes) & np.isin(edges[:, 1], nodes)
        local_edges = local_map[edges[edge_keep]]
        local_edge_index = _sym_edges(local_edges)
        import torch

        pe = _window_laplacian_pos_enc(
            torch.from_numpy(local_edge_index).long(), len(nodes), DEFAULT_POS_DIM
        ).numpy()
        members.append(nodes)
        lpe.append(pe)
        member_offsets.append(member_offsets[-1] + len(nodes))
        radii[i] = radius
        valid[i] = True
        n_intermediate[i] = int(len(nodes) - k)

    # The loop above must add one offset per point, including pre-cutoff rejects.
    # Fill gaps deterministically by rebuilding the ragged arrays in point order.
    member_offsets = [0]
    ordered_members, ordered_lpe = [], []
    built = iter(zip(members, lpe, strict=True))
    for ok in valid:
        if ok:
            nodes, pe = next(built)
            ordered_members.append(nodes); ordered_lpe.append(pe)
            member_offsets.append(member_offsets[-1] + len(nodes))
        else:
            member_offsets.append(member_offsets[-1])
    return {
        "synapse_id": points["synapse_id"].to_numpy(np.int64),
        "synapse_xyz_nm": xyz.astype(np.float32),
        "nearest_observed_node": centers,
        "nearest_observed_distance_nm": nearest_dist.astype(np.float32),
        "within_match_cutoff": cutoff_ok,
        "valid_k_window": valid,
        "radius_nm": radii,
        "n_intermediate": n_intermediate,
        "window_offsets": np.asarray(member_offsets, np.int64),
        "window_members": np.concatenate(ordered_members) if ordered_members else np.zeros(0, np.int32),
        "window_lpe": np.concatenate(ordered_lpe) if ordered_lpe else np.zeros((0, DEFAULT_POS_DIM), np.float32),
    }


def build_cell(root_id: int, points: pd.DataFrame, out: Path, k: int) -> dict:
    import torch

    graph = torch.load(GRAPH_CACHE / f"{root_id}.pt", map_location="cpu", weights_only=False)
    with open(CAVE_CACHE / f"{root_id}.pkl", "rb") as f:
        cave = pickle.load(f)
    with np.load(NEW_SKELETONS / f"{root_id}.npz") as z:
        new_pos = z["vertices"].copy()
        new_edges = z["edges"].copy()

    nuclei = json.loads(NUCLEI.read_text())["positions"]
    nucleus = np.asarray(nuclei[str(root_id)], np.float64)
    root_old = int(cKDTree(cave.coords).query(nucleus, k=1)[1])
    root_xyz = cave.coords[root_old].astype(np.float64)

    ck, crem, cpos, cedges, cweight, _ = _induced_cut(cave.coords, cave.edges, root_xyz)
    nk, _, npos, nedges, nweight, _ = _induced_cut(new_pos, new_edges, root_xyz)
    orig = graph.orig_node_ids.numpy().astype(np.int64)
    emb = graph.x.numpy().astype(np.float32)
    emb_keep = ck[orig]
    cave_ids = crem[orig[emb_keep]]
    emb = emb[emb_keep]

    cave_observed = np.zeros(len(cpos), bool)
    cave_observed[cave_ids] = True

    map_dist, map_node = cKDTree(npos).query(cpos[cave_ids], k=1)
    map_ok = map_dist <= EMBEDDING_MAP_CUTOFF_NM
    new_count = np.zeros(len(npos), np.int32)
    accepted_node = map_node[map_ok].astype(np.int64)
    accepted_x = emb[map_ok]
    order = np.argsort(accepted_node, kind="stable")
    accepted_node, accepted_x = accepted_node[order], accepted_x[order]
    new_observed_ids, first, counts = np.unique(
        accepted_node, return_index=True, return_counts=True
    )
    # Reduce only the accepted rows.  A dense new_nodes x 64 accumulator can
    # exceed RAM for the largest skeletons, most of whose nodes are unobserved.
    if len(new_observed_ids):
        summed = np.add.reduceat(accepted_x.astype(np.float64), first, axis=0)
        new_observed_x = (summed / counts[:, None]).astype(np.float32)
        new_count[new_observed_ids] = counts.astype(np.int32)
    else:
        new_observed_x = np.zeros((0, emb.shape[1]), np.float32)
    new_observed = new_count > 0
    new_observed_ids = new_observed_ids.astype(np.int32)

    cave_windows = _window_arrays(points, cpos, cedges, cweight, cave_observed, k)
    new_windows = _window_arrays(points, npos, nedges, nweight, new_observed, k)
    arrays = {
        "root_id": np.asarray([root_id], np.uint64),
        "root_cave_node_original": np.asarray([root_old], np.int64),
        "root_xyz_nm": root_xyz.astype(np.float32),
        "k_observed": np.asarray([k], np.int32),
        "cave_pos_nm": cpos, "cave_edges": cedges, "cave_edge_length_nm": cweight,
        # Features are sparse by node id.  A dense (new_nodes x 64) matrix
        # would waste hundreds of GB on unobserved intermediate nodes.
        "cave_observed_node_ids": cave_ids.astype(np.int32),
        "cave_observed_x": emb, "cave_observed": cave_observed,
        "new_pos_nm": npos, "new_edges": nedges, "new_edge_length_nm": nweight,
        "new_observed_node_ids": new_observed_ids,
        "new_observed_x": new_observed_x, "new_observed": new_observed,
        "embedding_to_new_distance_nm": map_dist.astype(np.float32),
        "embedding_to_new_within_match_cutoff": map_ok,
        "embedding_to_new_node": map_node.astype(np.int32),
        "embedding_to_new_collision_count": new_count,
    }
    arrays.update({f"cave_{name}": value for name, value in cave_windows.items()})
    arrays.update({f"new_{name}": value for name, value in new_windows.items()})
    _atomic_savez(out / "cells" / f"{root_id}.npz", **arrays)
    return {
        "root_id": root_id, "n_presynaptic": len(points),
        "cave_nodes": len(cpos), "new_nodes": len(npos),
        "input_embeddings": len(emb), "mapped_embeddings": int(map_ok.sum()),
        "new_observed_nodes": int(new_observed.sum()),
        "cave_valid_windows": int(cave_windows["valid_k_window"].sum()),
        "new_valid_windows": int(new_windows["valid_k_window"].sum()),
    }


def build(out: Path, task_id: int, num_tasks: int, k: int, force: bool) -> None:
    cohort = json.loads((out / "proofread_axon_root_ids.json").read_text())["root_ids"]
    manifest = json.loads(MANIFEST.read_text())["cells"]
    nuclei = json.loads(NUCLEI.read_text())["positions"]
    presyn = pd.read_parquet(PRESYNAPTIC)
    grouped = {int(r): g for r, g in presyn.groupby("cell_root_id", sort=False)}
    eligible = sorted(set(cohort) & {int(r) for r in manifest})
    selected = eligible[task_id::num_tasks]
    status_path = out / "status" / f"task_{task_id:04d}.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, rid in enumerate(selected, 1):
        dest = out / "cells" / f"{rid}.npz"
        reason = None
        required = [GRAPH_CACHE / f"{rid}.pt", CAVE_CACHE / f"{rid}.pkl", NEW_SKELETONS / f"{rid}.npz"]
        if str(rid) not in nuclei:
            reason = "missing CAVE root/nucleus position"
        elif rid not in grouped or len(grouped[rid]) == 0:
            reason = "no presynaptic points"
        elif any(not p.exists() for p in required):
            reason = "missing input: " + ", ".join(str(p) for p in required if not p.exists())
        try:
            if reason:
                row = {"root_id": rid, "status": "excluded", "reason": reason}
            elif dest.exists() and not force:
                row = {"root_id": rid, "status": "cached"}
            else:
                row = {"status": "ok", **build_cell(rid, grouped[rid], out, k)}
        except Exception as exc:  # per-cell failures remain visible and resumable
            row = {"root_id": rid, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(f"[{i}/{len(selected)}] {rid}: {row['status']} {row.get('reason', '')}", flush=True)
    tmp = status_path.with_name(status_path.name + f".tmp.{os.getpid()}")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, status_path)


def _plot(values: np.ndarray, title: str, xlabel: str, out: Path,
          rejected_label: str, cutoff_nm: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = values[np.isfinite(values)] / 1000.0
    rejected = float(np.mean(finite > cutoff_nm / 1000.0)) if len(finite) else np.nan
    xmax = max(2.2, min(float(np.percentile(finite, 99.5)), 20.0)) if len(finite) else 2.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finite, bins=np.linspace(0, xmax, 101), color="#3978b5", alpha=0.85)
    cutoff_um = cutoff_nm / 1000.0
    ax.axvline(cutoff_um, color="#b22222", lw=2, label=f"{cutoff_um:g} µm cutoff")
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    ax.text(.98, .95, f"{rejected_label}: {rejected:.2%}", transform=ax.transAxes,
            ha="right", va="top", bbox={"facecolor": "white", "alpha": .9})
    ax.legend(); fig.tight_layout()
    fig.savefig(out, dpi=180); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def finalize(out: Path) -> None:
    files = sorted((out / "cells").glob("*.npz"))
    map_d, cave_d, new_d, inventory = [], [], [], []
    for path in files:
        with np.load(path) as z:
            map_d.append(z["embedding_to_new_distance_nm"])
            cave_d.append(z["cave_nearest_observed_distance_nm"])
            new_d.append(z["new_nearest_observed_distance_nm"])
            inventory.append({
                "root_id": int(z["root_id"][0]),
                "n_presynaptic": len(z["cave_synapse_id"]),
                "input_embeddings": len(z["embedding_to_new_distance_nm"]),
                "mapped_embeddings": int(z["embedding_to_new_within_match_cutoff"].sum()),
                "cave_valid_windows": int(z["cave_valid_k_window"].sum()),
                "new_valid_windows": int(z["new_valid_k_window"].sum()),
            })
    concat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.float32)
    map_d, cave_d, new_d = map(concat, (map_d, cave_d, new_d))
    status = []
    for path in sorted((out / "status").glob("task_*.jsonl")):
        status.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    proofread_ids = set(json.loads((out / "proofread_axon_root_ids.json").read_text())["root_ids"])
    current_ids = {int(r) for r in json.loads(MANIFEST.read_text())["cells"]}
    eligible_ids = proofread_ids & current_ids
    built_ids = {x["root_id"] for x in inventory}
    meta = {
        "format": "presynaptic-axon-npz-v1", "created_from": str(ROOT),
        "materialization_version": MAT_VERSION, "proofreading_predicate": "status_axon == True",
        "soma_cut_center": "CAVE node nearest cells.soma_{x,y,z}_nm",
        "soma_radius_nm": SOMA_RADIUS_NM,
        "embedding_map_cutoff_nm": EMBEDDING_MAP_CUTOFF_NM,
        "synapse_match_cutoff_nm": SYNAPSE_MATCH_CUTOFF_NM,
        "k_observed": DEFAULT_K, "lpe_dim": DEFAULT_POS_DIM,
        "n_proofread_axon_roots_v1718": len(proofread_ids),
        "n_with_current_embeddings": len(eligible_ids),
        "n_discarded_without_current_embeddings": len(proofread_ids - current_ids),
        "discarded_without_current_embeddings_root_ids": sorted(proofread_ids - current_ids),
        "n_cells": len(files),
        "missing_built_root_ids": sorted(eligible_ids - built_ids),
        "n_presynaptic": int(sum(x["n_presynaptic"] for x in inventory)),
        "n_input_embeddings": len(map_d),
        "n_mapped_embeddings": int((map_d <= EMBEDDING_MAP_CUTOFF_NM).sum()),
        "embedding_mapping_discard_fraction": float(np.mean(map_d > EMBEDDING_MAP_CUTOFF_NM)),
        "n_cave_presynaptic_within_match_cutoff": int((cave_d <= SYNAPSE_MATCH_CUTOFF_NM).sum()),
        "cave_presynaptic_cutoff_discard_fraction": float(np.mean(cave_d > SYNAPSE_MATCH_CUTOFF_NM)),
        "n_new_presynaptic_within_match_cutoff": int((new_d <= SYNAPSE_MATCH_CUTOFF_NM).sum()),
        "new_presynaptic_cutoff_discard_fraction": float(np.mean(new_d > SYNAPSE_MATCH_CUTOFF_NM)),
        "n_cave_valid_windows": int(sum(x["cave_valid_windows"] for x in inventory)),
        "n_new_valid_windows": int(sum(x["new_valid_windows"] for x in inventory)),
        "status_counts": dict(Counter(x["status"] for x in status)),
        "arrays": {
            "*_pos_nm": "node xyz, nm",
            "*_observed_node_ids + *_observed_x": "sparse node-id to SegCLR-embedding table",
            "*_window_members": "ragged node IDs indexed by *_window_offsets",
            "*_window_lpe": "row-aligned with *_window_members; window-local Laplacian PE",
        },
        "inventory": inventory,
    }
    _atomic_json(out / "metadata.json", meta)
    fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    _plot(map_d, "SegCLR CAVE node → nearest new-skeleton node", "nearest distance (µm)",
          fig / "embedding_to_new_skeleton_distance.png", "embeddings discarded",
          EMBEDDING_MAP_CUTOFF_NM)
    _plot(cave_d, "Presynaptic point → nearest CAVE SegCLR node", "nearest distance (µm)",
          fig / "presynaptic_to_cave_segclr_distance.png", "presynaptic points beyond cutoff",
          SYNAPSE_MATCH_CUTOFF_NM)
    _plot(new_d, "Presynaptic point → nearest mapped new-skeleton SegCLR node", "nearest distance (µm)",
          fig / "presynaptic_to_new_segclr_distance.png", "presynaptic points beyond cutoff",
          SYNAPSE_MATCH_CUTOFF_NM)
    print(json.dumps({k: v for k, v in meta.items() if k not in {"inventory", "arrays"}}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.k != DEFAULT_K:
        raise SystemExit(f"this database is fixed at K={DEFAULT_K}; got --k={args.k}")
    if args.prepare:
        prepare(args.out, os.environ.get("CAVE_TOKEN", ""))
    elif args.build:
        build(args.out, args.task_id, args.num_tasks, args.k, args.force)
    else:
        finalize(args.out)


if __name__ == "__main__":
    main()
