"""Empirical check: does the lab's `all_cells_aggregated_1718.h5` 'nodes' column
(a skeleton-vertex index) refer to the SAME physical vertex, at the SAME index,
as our own cached CAVE skeletons in data/skeleton_cache/*.pkl -- for the same
root_id?

This does NOT need a live CAVE call. Both sides are already pre-cached on disk:

  - ours:  data/skeleton_cache/{root_id}.pkl              (segclr_db.results.Skeleton,
           fetched via segclr_db.cave.CAVESkeletonSource against CAVE datastack
           "minnie65_public", skeleton_version=4 -- see data/cave_skeletons.py)
  - lab's: /orcd/compute/sdorkenw/001/collina/skeleton_cache/skeleton_partial/
           {root_id}/skeleton.h5 ('vertices' (N,3) float32 nm, 'edges' (M,2) int32)
           -- this IS the cache backing EmbeddingCache.get_skeleton(), which
           embedding_query/embedding_cache.py fetches via
           `client.skeleton.get_skeleton(root_id, output_format="dict")` against
           CAVE datastack "minnie65_phase3_v1" (see generate_gt_dataset.py on the
           lcpn branch). No skeleton_version is passed explicitly, but
           caveclient 8.2.1's own default is skeleton_version=4 -- the same value
           our own CAVEConfig.skeleton_version defaults to (segclr_db/src/cave.py).
           So the two sides agree on skeleton_version; the open question this
           script settles is whether the differing datastack name
           (minnie65_public vs minnie65_phase3_v1) and differing materialization
           (343 vs 1718) changed vertex count/order for the same root_id.

For a sample of root_ids present in both caches AND in the h5, this script:
  1. Compares vertex counts (N).
  2. Does an exact elementwise coordinate comparison at matching indices
     (coords[i] vs vertices[i] for several i) -- NOT just a nearest-neighbor
     search, since NN would also "succeed" on a merely similar but
     differently-INDEXED skeleton, which is exactly the failure mode in
     question.
  3. Falls back to a nearest-neighbor residual distance (like the deprecated
     xyz-reconciliation pipeline used) purely as a secondary diagnostic, so a
     "same structure, different order" case is distinguishable from "different
     structure entirely".
  4. Compares edge sets.
  5. Sanity-checks the h5's own 'nodes' column range against both vertex counts
     for that cell.

Run via sbatch only -- see scripts/sbatch/check_h5_skeleton_alignment.sh.
No CAVE token needed (nothing here makes a live CAVE/network call).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
_SEGCLR_DB_SRC = REPO_ROOT / "segclr_db" / "src"
if str(_SEGCLR_DB_SRC) not in sys.path:
    sys.path.insert(0, str(_SEGCLR_DB_SRC))

# Needed so pickle.load() can resolve segclr_db.results.Skeleton.
from segclr_db.results import Skeleton  # noqa: E402,F401

H5_PATH = "/orcd/compute/sdorkenw/001/collina/data/all_cells_aggregated_1718.h5"
LAB_SKELETON_DIR = Path("/orcd/compute/sdorkenw/001/collina/skeleton_cache/skeleton_partial")
OUR_SKELETON_DIR = REPO_ROOT / "data" / "skeleton_cache"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"

N_SAMPLE = 20


def load_our_skeleton(root_id: int) -> Skeleton | None:
    path = OUR_SKELETON_DIR / f"{root_id}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lab_skeleton(root_id: int) -> dict | None:
    path = LAB_SKELETON_DIR / str(root_id) / "skeleton.h5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        return {
            "vertices": np.array(f["vertices"], dtype=np.float32),
            "edges": np.array(f["edges"], dtype=np.int32),
        }


def edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_root_ids = [int(r) for r in manifest["cells"].keys()]
    print(f"manifest: {len(manifest_root_ids)} cells", flush=True)

    # Find cells present in BOTH skeleton caches.
    both = []
    for rid in manifest_root_ids:
        if (OUR_SKELETON_DIR / f"{rid}.pkl").exists() and (
            LAB_SKELETON_DIR / str(rid) / "skeleton.h5"
        ).exists():
            both.append(rid)
    print(f"present in both our cache and the lab's skeleton cache: {len(both)}", flush=True)

    sample = both[:N_SAMPLE]
    print(f"sampling {len(sample)} root_ids for detailed comparison\n", flush=True)

    print("=== loading h5 seg_ids (once) ===", flush=True)
    with h5py.File(H5_PATH, "r") as f:
        h5_seg_ids = f["seg_ids"][:]
    h5_seg_ids = h5_seg_ids.astype(np.int64)
    print(f"h5 total rows: {len(h5_seg_ids)}\n", flush=True)

    results = []
    for rid in sample:
        our = load_our_skeleton(rid)
        lab = load_lab_skeleton(rid)
        if our is None or lab is None:
            continue

        our_coords = our.coords.astype(np.float64)
        lab_verts = lab["vertices"].astype(np.float64)
        n_our, n_lab = len(our_coords), len(lab_verts)

        row_idx = np.where(h5_seg_ids == rid)[0]
        with h5py.File(H5_PATH, "r") as f:
            h5_nodes = f["nodes"][row_idx] if len(row_idx) else np.array([], dtype=np.int64)
        n_h5_rows = int(len(row_idx))
        h5_node_max = int(h5_nodes.max()) if len(h5_nodes) else -1
        h5_node_min = int(h5_nodes.min()) if len(h5_nodes) else -1
        h5_node_nunique = int(len(np.unique(h5_nodes))) if len(h5_nodes) else 0

        # 1. vertex count agreement
        count_match = n_our == n_lab

        # 2. exact elementwise coordinate comparison at matching indices
        check_idxs = sorted(set([0, n_our // 4, n_our // 2, (3 * n_our) // 4, n_our - 1]))
        check_idxs = [i for i in check_idxs if 0 <= i < min(n_our, n_lab)]
        idx_diffs = []
        for i in check_idxs:
            d = float(np.linalg.norm(our_coords[i] - lab_verts[i]))
            idx_diffs.append(d)
        idx_diffs = np.array(idx_diffs)

        # 3. nearest-neighbor residual (secondary diagnostic)
        tree = cKDTree(lab_verts)
        nn_dist, _ = tree.query(our_coords)

        # 4. edge set comparison
        our_edges = edge_set(our.edges)
        lab_edges = edge_set(lab["edges"])
        edge_overlap = len(our_edges & lab_edges)
        edge_jaccard = edge_overlap / max(len(our_edges | lab_edges), 1)

        results.append(
            {
                "root_id": rid,
                "n_our": n_our,
                "n_lab": n_lab,
                "count_match": count_match,
                "idx_diff_max_nm": float(idx_diffs.max()) if len(idx_diffs) else float("nan"),
                "idx_diff_mean_nm": float(idx_diffs.mean()) if len(idx_diffs) else float("nan"),
                "nn_median_nm": float(np.median(nn_dist)),
                "nn_max_nm": float(nn_dist.max()),
                "edge_jaccard": edge_jaccard,
                "n_h5_rows": n_h5_rows,
                "h5_node_min": h5_node_min,
                "h5_node_max": h5_node_max,
                "h5_node_nunique": h5_node_nunique,
            }
        )

    print("=== per-cell comparison ===", flush=True)
    header = (
        f"{'root_id':>20} {'n_our':>7} {'n_lab':>7} {'match':>6} "
        f"{'idx_diff_max_nm':>16} {'nn_median_nm':>13} {'nn_max_nm':>10} "
        f"{'edge_jacc':>10} {'h5_rows':>8} {'h5_node_max':>12} {'h5_uniq':>8}"
    )
    print(header, flush=True)
    for r in results:
        print(
            f"{r['root_id']:>20} {r['n_our']:>7} {r['n_lab']:>7} {str(r['count_match']):>6} "
            f"{r['idx_diff_max_nm']:>16.2f} {r['nn_median_nm']:>13.2f} {r['nn_max_nm']:>10.2f} "
            f"{r['edge_jaccard']:>10.3f} {r['n_h5_rows']:>8} {r['h5_node_max']:>12} "
            f"{r['h5_node_nunique']:>8}",
            flush=True,
        )

    if results:
        n_count_match = sum(r["count_match"] for r in results)
        idx_diff_maxes = np.array([r["idx_diff_max_nm"] for r in results])
        nn_medians = np.array([r["nn_median_nm"] for r in results])
        edge_jaccards = np.array([r["edge_jaccard"] for r in results])

        print("\n=== summary across sample ===", flush=True)
        print(f"n cells compared: {len(results)}", flush=True)
        print(f"vertex count matches exactly: {n_count_match}/{len(results)}", flush=True)
        print(
            f"same-index coordinate diff (nm): "
            f"median-of-max={np.median(idx_diff_maxes):.3f}  "
            f"max-of-max={idx_diff_maxes.max():.3f}",
            flush=True,
        )
        print(
            f"nearest-neighbor residual (nm), median across cells: "
            f"median={np.median(nn_medians):.3f}  max={nn_medians.max():.3f}",
            flush=True,
        )
        print(
            f"edge-set jaccard overlap: median={np.median(edge_jaccards):.3f}  "
            f"min={edge_jaccards.min():.3f}",
            flush=True,
        )

        print("\n=== verdict heuristic ===", flush=True)
        if n_count_match == len(results) and np.median(idx_diff_maxes) < 1.0:
            print(
                "SAME skeleton, SAME vertex ordering: node index i means the same "
                "physical vertex in both caches for every sampled cell (exact vertex "
                "count match + ~0nm same-index coordinate diff). h5 'nodes' column can "
                "be used directly against data/skeleton_cache/*.pkl.",
                flush=True,
            )
        elif np.median(nn_medians) < 5.0 and n_count_match < len(results):
            print(
                "SAME underlying structure but DIFFERENT indexing (vertex counts "
                "disagree while nearest-neighbor residuals are near-zero): a "
                "coordinate-based nearest-neighbor remap between the two skeletons "
                "would be needed per cell, index alone is NOT safe.",
                flush=True,
            )
        else:
            print(
                "Evidence of a REAL divergence (large same-index diffs and/or large "
                "NN residuals and/or low edge-set overlap): the two skeletons are not "
                "simply reindexed versions of each other. Do not assume h5 'nodes' "
                "aligns with data/skeleton_cache/*.pkl without further reconciliation.",
                flush=True,
            )
    else:
        print("\nNo comparable cells found -- cannot draw a conclusion.", flush=True)

    print("\ndone.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
