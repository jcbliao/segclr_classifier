"""Match-quality validation for raw v943 embeddings snapped onto OUR cached
skeleton (data/skeleton_cache/*.pkl) -- before committing to fetching all
2442 cells, check on a small sample whether the snap distances look sane,
the same rigor CLAUDE.md documents for the original v343 validation (median
740nm / mean 893nm / p95 2192nm / max 7206nm on one cell, relative to local
process radius, not cell span).

Unlike that original validation, we're not matching against an independently
fetched/uncertain skeleton -- data/skeleton_cache/*.pkl is confirmed
(scripts/check_h5_skeleton_alignment.py) to be exactly the same skeleton the
lab's own pipeline uses to build all_cells_aggregated_1718.h5, including
their own raw-v943-to-skeleton snap. So this is really asking "does OUR
cKDTree snap, run independently, land close to where theirs presumably did"
-- not validating a previously-unvalidated skeleton source.

Run via sbatch (mit_normal -- network I/O to GCS, not GPU work).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import cave_skeletons as cs  # noqa: E402
from data.dataset_lcpn import load_manifest  # noqa: E402
from data.public_reader import get_public_filesystem, get_raw_cell_embeddings  # noqa: E402

N_SAMPLE_CELLS = 15


def main():
    manifest = load_manifest()
    val_root_ids = [
        int(rid) for rid, info in manifest["cells"].items()
        if info["split"] == "val" and info.get("has_graph", False)
    ]
    rng = np.random.default_rng(0)
    sample = rng.choice(val_root_ids, size=min(N_SAMPLE_CELLS, len(val_root_ids)), replace=False)
    print(f"sampling {len(sample)} cells from val split", flush=True)

    fs = get_public_filesystem()

    all_dists = []
    all_pts_per_cell = []
    all_nodes_per_cell = []
    for root_id in sample:
        root_id = int(root_id)
        skeleton = cs.load_cached(root_id)
        if skeleton is None:
            print(f"  cell {root_id}: no cached skeleton, skipping", flush=True)
            continue

        raw = get_raw_cell_embeddings(root_id, filesystem=fs, data_key="microns_v943")
        n_pts = raw.xyz_nm.shape[0]
        if n_pts == 0:
            print(f"  cell {root_id}: 0 raw v943 embeddings returned", flush=True)
            continue

        tree = cKDTree(skeleton.coords)
        snap_dists, node_idx = tree.query(raw.xyz_nm, k=1)

        all_dists.append(snap_dists)
        all_pts_per_cell.append(n_pts)
        all_nodes_per_cell.append(len(skeleton))
        n_nodes_hit = len(np.unique(node_idx))
        print(
            f"  cell {root_id}: {n_pts} raw pts, {len(skeleton)} skeleton nodes, "
            f"{n_nodes_hit} nodes with >=1 snap, "
            f"median_dist={np.median(snap_dists):.0f}nm mean={snap_dists.mean():.0f}nm "
            f"p95={np.percentile(snap_dists, 95):.0f}nm max={snap_dists.max():.0f}nm",
            flush=True,
        )

    if all_dists:
        all_dists_cat = np.concatenate(all_dists)
        print(f"\n=== aggregate across {len(all_dists)} cells, {len(all_dists_cat)} points ===", flush=True)
        print(f"median={np.median(all_dists_cat):.0f}nm  mean={all_dists_cat.mean():.0f}nm", flush=True)
        print(
            f"p50={np.percentile(all_dists_cat, 50):.0f}nm "
            f"p95={np.percentile(all_dists_cat, 95):.0f}nm "
            f"p99={np.percentile(all_dists_cat, 99):.0f}nm "
            f"max={all_dists_cat.max():.0f}nm",
            flush=True,
        )
        print(
            f"avg raw points per cell: {np.mean(all_pts_per_cell):.0f}, "
            f"avg skeleton nodes per cell: {np.mean(all_nodes_per_cell):.0f}, "
            f"ratio: {np.mean(all_pts_per_cell) / np.mean(all_nodes_per_cell):.2f}",
            flush=True,
        )

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
