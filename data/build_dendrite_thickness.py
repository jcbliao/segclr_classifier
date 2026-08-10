"""Bulk dendrite-thickness ingestion: for every cell already in
data/graph_cache/*.pt, compute the spine-corrected shaft radius
(data/dendrite_thickness.py) at every eligible skeleton vertex, using ONLY
small local mesh patches (data/neuron_mesh.py) -- never a whole-neuron mesh
download, per explicit user direction.

Batching, not one mesh fetch per vertex: eligible vertices (dendrite
compartment, non-branch-point, finite tangent) are grouped into spatial
buckets (an axis-aligned grid, BUCKET_SIZE_NM per side) so nearby vertices
along the same stretch of cable share ONE local mesh-patch fetch and one
ray-casting call, rather than hitting CAVE once per vertex. At ~1.8um
skeleton spacing, a 15um bucket holds several vertices typically -- cuts
request count by roughly that factor versus per-vertex fetching, while each
patch (bucket extent + margin) stays a small local region
(scripts/check_local_mesh_patch.py measured ~5.7k vertices for a ~12um-wide
patch -- orders of magnitude below a multi-million-face whole neuron mesh).

Alignment: skeleton (vertices/edges/compartments/radii) comes from
data/skeleton_cache/*.pkl -- the SAME Skeleton object build_dataset_from_store.py
already used to build data/graph_cache/*.pt's `pos`/`x`, so no separate
position-based re-matching is needed there (data.orig_node_ids already gives
the exact index back into this same skeleton -- see
data/dendrite_thickness.py's module docstring). Only the mesh is new.

Output: data/dendrite_thickness_cache/{root_id}.npz, radius_nm (n_skeleton_vertices,)
float32 aligned to skeleton.coords (NaN where unmeasured -- non-dendrite,
branch point, or a mesh-hole miss), same indexing data/geodesic_window.py
already uses for pos/compartments. Resumable: cells with a cached .npz are
skipped.

Run via sbatch as a SLURM array (mit_preemptable -- CAVE-network-bound, real
but unknown per-cell request count across 2192 cells, same "long pole"
precedent as skeleton ingestion, see CLAUDE.md), scripts/sbatch/build_dendrite_thickness.sh,
per explicit user direction (2026-08-07): each array task owns a fixed,
deterministic SHARD of cells (--shard-index/--num-shards below, weighted by
manifest n_nodes_covered so shards are roughly balanced by expected work, not
just cell count -- same "split by cumulative node count, not cell count"
principle CLAUDE.md's scale-characteristics section already established for
the analogous Lance-write sharding problem). `#SBATCH --requeue` lets SLURM
automatically resubmit a task preempted mid-run; because every cell's result
is only written (np.savez) after it fully completes, a requeued task just
resumes -- it re-scans its OWN shard's still-todo cells and skips whatever
already finished, no partial/corrupt state possible.

    python data/build_dendrite_thickness.py --workers 8 --shard-index 0 --num-shards 16
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from data import cave_skeletons as cs  # noqa: E402
from data import dendrite_thickness as dt  # noqa: E402
from data import neuron_mesh as nm  # noqa: E402

THICKNESS_CACHE_DIR = Path(__file__).resolve().parent / "dendrite_thickness_cache"

# Bucket edge length for grouping nearby eligible vertices into one shared
# local mesh-patch fetch -- see module docstring for the request-count/
# patch-size trade-off. Margin added on top comes from neuron_mesh.DEFAULT_MARGIN_NM.
#
# Raised twice: 15_000 -> 40_000 (measured ~250s/cell at 15um; the full
# 2192-cell corpus at that rate would take days and a huge request volume
# against CAVE's shared mesh service) -> 80_000 (2891 conn-pool warnings /
# ~104s-per-cell average at 40um was still request-heavy -- see
# data/neuron_mesh.py's throughput-fix note for the connection-pool-size and
# fragment-cache halves of this same fix). 80um is still a negligible
# fraction of a whole neuron's extent (100s of um to >1mm) -- request count
# drops roughly in proportion to bucket size along a typically near-linear
# stretch of dendrite, at the cost of a somewhat larger (but still local)
# mesh fetch per bucket.
DEFAULT_BUCKET_SIZE_NM = 80_000.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("build_dendrite_thickness")


def _cache_path(root_id: int) -> Path:
    return THICKNESS_CACHE_DIR / f"{root_id}.npz"


def balanced_shards(root_ids: list[int], weights: dict[int, float], num_shards: int) -> list[list[int]]:
    """Deterministic, work-balanced partition of root_ids into num_shards
    groups -- one SLURM array task's fixed cell set. Greedy longest-
    processing-time-first: sort cells by DESCENDING weight (heaviest first)
    and always drop the next cell onto whichever shard currently has the
    least accumulated weight. This is the standard approximation for
    balanced multiway partitioning (no exact solution is needed here, just
    "no shard is dramatically heavier than another").

    weights: {root_id: n_nodes_covered} (manifest.json) -- a proxy for
    expected ray-casting work per cell (more covered nodes -> more eligible
    dendrite vertices -> more buckets -> more mesh fetches), same rationale
    CLAUDE.md's scale-characteristics section already uses for Lance-write
    sharding. Ties broken by root_id for full determinism (Python's sort is
    stable, but root_ids is already sorted by the caller, so this only
    matters if it isn't).

    Deliberately partitions ALL of root_ids, not just the still-todo subset
    -- so shard membership never shifts as cells complete across separate
    runs/requeues of the same array (task N always means the same cells).
    """
    order = sorted(root_ids, key=lambda r: (-weights.get(r, 0), r))
    shards: list[list[int]] = [[] for _ in range(num_shards)]
    shard_weight = [0.0] * num_shards
    for root_id in order:
        i = min(range(num_shards), key=lambda i: shard_weight[i])
        shards[i].append(root_id)
        shard_weight[i] += weights.get(root_id, 0)
    return shards


def _spatial_buckets(coords_nm: np.ndarray, indices: np.ndarray, bucket_size_nm: float) -> dict:
    """{bucket_key: [skeleton indices]} -- axis-aligned grid over just the
    given (already-eligible) indices. Deterministic order (sorted keys) so
    output is reproducible run to run."""
    keys = np.floor(coords_nm[indices] / bucket_size_nm).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx, key in zip(indices, map(tuple, keys)):
        buckets.setdefault(key, []).append(int(idx))
    return dict(sorted(buckets.items()))


def compute_one_cell(
    root_id: int, cv, bucket_size_nm: float = DEFAULT_BUCKET_SIZE_NM
) -> tuple[np.ndarray, dict]:
    """Returns (radius_nm (n_skeleton_vertices,) float32, stats dict).

    stats: n_eligible, n_measured, n_buckets, n_empty_patches -- logged per
    cell so a systematically-failing cell (e.g. all patches empty, suggesting
    a coordinate-frame problem) is visible immediately rather than silently
    producing an all-NaN cache file that looks identical to "genuinely no
    dendrite vertices".
    """
    with open(cs._cache_path(root_id), "rb") as f:
        skel = pickle.load(f)

    rc_skel = dt.skeleton_for_ray_casting(skel)
    tangents, _ = dt.local_tangents(rc_skel)
    _, _, degree = dt.skeleton_neighbors(rc_skel)

    is_dendrite = np.isin(skel.compartments, dt.DENDRITE_COMPARTMENTS)
    eligible = np.isfinite(tangents).all(axis=1) & is_dendrite & (degree <= 2)
    eligible_idx = np.where(eligible)[0]

    radius_nm = np.full(len(skel.coords), np.nan, dtype=np.float32)
    stats = {"n_eligible": len(eligible_idx), "n_measured": 0, "n_buckets": 0, "n_empty_patches": 0}
    if len(eligible_idx) == 0:
        return radius_nm, stats

    buckets = _spatial_buckets(skel.coords.astype(np.float64), eligible_idx, bucket_size_nm)
    stats["n_buckets"] = len(buckets)

    for bucket_indices in buckets.values():
        idx = np.asarray(bucket_indices, dtype=np.int64)
        points_nm = skel.coords[idx].astype(np.float64)
        mesh = nm.fetch_local_mesh_patch(cv, root_id, points_nm)
        if mesh is None:
            stats["n_empty_patches"] += 1
            continue

        vertex_mask = np.zeros(len(skel.coords), dtype=bool)
        vertex_mask[idx] = True
        try:
            df = dt.estimate_dendrite_radius(
                mesh, rc_skel, vertex_mask=vertex_mask, tangents=tangents, exclude_branch_points=True
            )
        except ValueError:
            # "No eligible vertices" -- every point in this bucket happened
            # to land at a spot with a degenerate/missing tangent after all;
            # already NaN in the output, nothing to do.
            continue

        radius_nm[df.index.to_numpy()] = df["radius_nm"].to_numpy(dtype=np.float32)
        stats["n_measured"] += int(np.isfinite(df["radius_nm"]).sum())

    return radius_nm, stats


def main(args) -> int:
    THICKNESS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    import json

    manifest = json.loads(manifest_path.read_text())
    root_ids = sorted(int(r) for r in manifest["cells"])

    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise SystemExit(f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}")
        weights = {r: manifest["cells"][str(r)].get("n_nodes_covered", 0) for r in root_ids}
        shard_root_ids = balanced_shards(root_ids, weights, args.num_shards)[args.shard_index]
        logger.info(
            "shard %d/%d: %d of %d total cells (weight-balanced by n_nodes_covered)",
            args.shard_index, args.num_shards, len(shard_root_ids), len(root_ids),
        )
    else:
        shard_root_ids = root_ids

    todo = [r for r in shard_root_ids if not _cache_path(r).exists() and cs._cache_path(r).exists()]
    n_done = sum(1 for r in shard_root_ids if _cache_path(r).exists())
    n_no_skeleton = sum(1 for r in shard_root_ids if not cs._cache_path(r).exists())
    logger.info(
        "this shard: %d cells, %d already done, %d to do, %d skipped (no cached skeleton)",
        len(shard_root_ids), n_done, len(todo), n_no_skeleton,
    )
    if args.limit:
        todo = todo[: args.limit]
        logger.info("--limit %d: restricting this run to %d cells", args.limit, len(todo))
    if not todo:
        return 0

    token = os.environ.get("CAVE_TOKEN")
    if not token:
        raise SystemExit("CAVE_TOKEN not set -- see scripts/sbatch/build_dendrite_thickness.sh")
    client = cs.default_cave_config(token).build_client()
    cv = nm.local_cloudvolume(client)

    def _worker(root_id):
        t0 = time.monotonic()
        try:
            radius_nm, stats = compute_one_cell(root_id, cv, bucket_size_nm=args.bucket_size_nm)
        except Exception as exc:  # noqa: BLE001 -- reported per-cell, loop continues
            return root_id, None, None, f"{type(exc).__name__}: {exc}"
        return root_id, radius_nm, stats, time.monotonic() - t0

    n_ok, n_err = 0, 0
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, r): r for r in todo}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="cells", unit="cell")
        for fut in pbar:
            root_id, radius_nm, stats, err_or_elapsed = fut.result()
            if radius_nm is None:
                n_err += 1
                logger.warning("  cell %d: %s", root_id, err_or_elapsed)
                pbar.set_postfix(ok=n_ok, err=n_err)
                continue

            np.savez(_cache_path(root_id), radius_nm=radius_nm)
            n_ok += 1
            pbar.set_postfix(ok=n_ok, err=n_err)
            if stats["n_eligible"] and not stats["n_measured"]:
                logger.warning(
                    "  cell %d: %d eligible vertices, 0 measured (%d buckets, %d empty patches) "
                    "-- possible coordinate-frame or mesh-generation problem",
                    root_id, stats["n_eligible"], stats["n_buckets"], stats["n_empty_patches"],
                )

    elapsed = time.monotonic() - t_start
    logger.info(
        "done: %d ok, %d errors, in %.0fs (%.2f cells/s)", n_ok, n_err, elapsed, n_ok / max(elapsed, 1e-9)
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--bucket-size-nm", type=float, default=DEFAULT_BUCKET_SIZE_NM)
    p.add_argument(
        "--limit", type=int, default=None,
        help="process at most this many (not-yet-done) cells this run -- for a cheap "
             "validation pass before committing to the full corpus",
    )
    p.add_argument(
        "--shard-index", type=int, default=0,
        help="which of --num-shards fixed, weight-balanced cell groups this run processes "
             "(SLURM array task index -- see scripts/sbatch/build_dendrite_thickness.sh, "
             "normally set from $SLURM_ARRAY_TASK_ID, not passed by hand)",
    )
    p.add_argument(
        "--num-shards", type=int, default=1,
        help="total number of shards (SLURM array size, $SLURM_ARRAY_TASK_COUNT) -- 1 means "
             "no sharding, process every cell in this one run",
    )
    raise SystemExit(main(p.parse_args()))
