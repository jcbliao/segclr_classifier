"""graph_cache -> data/mask_volume_cache/{root_id}.npz: mask voxel count per node.

For every node that has an embedding, counts how many voxels inside that node's SegCLR
input window belonged to the cell. See :mod:`data.mask_volume` for the window geometry
and why it is reproduced rather than approximated.

The masks themselves are never written anywhere -- each crop is read, summed, and
dropped. Output is 4 bytes per node, so the whole dataset is ~50 MB rather than the
~25 TiB the stored volumes would be.

Alignment
---------
``voxel_count[i]`` belongs to graph node ``i`` of ``graph_cache/{root_id}.pt``, and
``orig_node_ids[i]`` is that node's index into the full skeleton vertex array. Both are
stored, so a consumer can join either way and never has to guess.

This differs deliberately from ``dendrite_thickness_cache``, which is indexed by skeleton
vertex and therefore needs ``orig_node_ids`` applied at read time. Thickness is defined on
skeleton geometry; a mask volume is defined only where an embedding exists, since the
window is the thing the embedding was computed from. Storing the skeleton-length array
would be mostly sentinel.

Usage
-----
    # what would run, reading nothing
    python -u data/build_mask_volume.py --dry-run

    # one rank, a few cells, to measure throughput
    python -u data/build_mask_volume.py --limit 5

    # the real run, as a SLURM array (rank/world come from the environment)
    sbatch scripts/sbatch/build_mask_volume.sh

Re-running is how you resume: a cell whose npz already exists is skipped, so a preempted
rank loses at most the cell it was on. ``--redo`` overwrites instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.mask_volume import (  # noqa: E402
    BOX_SIZE,
    DEFAULT_CACHE_BYTES,
    DEFAULT_MAT_VERSION,
    DEFAULT_NUM_THREADS,
    VOXEL_VOLUME_NM3,
    MaskVolumeCounter,
    clipped_flags,
)

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"
GRAPH_CACHE_DIR = REPO / "data" / "graph_cache"
OUTPUT_DIR = REPO / "data" / "mask_volume_cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--graph-cache", type=Path, default=GRAPH_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--mat-version",
        type=int,
        default=DEFAULT_MAT_VERSION,
        help="materialization the root_ids belong to; selects the segmentation volume",
    )
    parser.add_argument("--box-size", type=int, default=BOX_SIZE)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--cache-bytes", type=int, default=DEFAULT_CACHE_BYTES)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the cell list, for a pilot run"
    )
    parser.add_argument(
        "--redo", action="store_true", help="recount cells that already have an npz"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the plan and this rank's share, count nothing",
    )
    return parser.parse_args()


def task_from_env() -> tuple[int, int]:
    """``(rank, world)`` from a job array or srun, defaulting to a solo run."""
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        return rank, int(count) if count else rank + 1
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]), int(os.environ.get("SLURM_NTASKS", 1))
    return 0, 1


def plan_shards(root_ids, node_counts, n_tasks: int) -> list[np.ndarray]:
    """Split cells into ``n_tasks`` contiguous blocks of similar total node count.

    Balanced by **nodes, not cells**: per-cell counts span 262 to 25,434 here, a 97x
    spread, so equal cell counts would leave the slowest rank setting the wall clock.
    Contiguous in ``root_id`` rather than strided, which keeps each rank's reads in a
    narrow spatial region of the volume and therefore in its chunk cache.
    """
    root_ids = np.asarray(root_ids, dtype=np.int64)
    node_counts = np.asarray(node_counts, dtype=np.int64)
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if not len(root_ids):
        return [np.empty(0, np.int64) for _ in range(n_tasks)]

    order = np.argsort(root_ids)
    root_ids, node_counts = root_ids[order], node_counts[order]

    cumulative = np.cumsum(node_counts)
    targets = cumulative[-1] * np.arange(1, n_tasks) / n_tasks
    boundaries = np.searchsorted(cumulative, targets, side="left")
    return list(np.split(root_ids, boundaries))


def load_cell_list(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(root_ids, n_nodes)`` from the manifest, sorted by root_id."""
    manifest = json.loads(manifest_path.read_text())
    cells = manifest["cells"]
    root_ids = np.array(sorted(int(r) for r in cells), dtype=np.int64)
    n_nodes = np.array(
        [int(cells[str(r)]["n_nodes_covered"]) for r in root_ids], dtype=np.int64
    )
    return root_ids, n_nodes


def count_one_cell(counter: MaskVolumeCounter, root_id: int, graph_path: Path) -> dict:
    """Every node of one cell. Returns the arrays to be saved."""
    data = torch.load(graph_path, weights_only=False)
    coords_nm = data.pos.numpy()
    orig_node_ids = data.orig_node_ids.numpy().astype(np.int64)

    if len(coords_nm) != len(orig_node_ids):
        raise ValueError(
            f"cell {root_id}: pos has {len(coords_nm)} rows but orig_node_ids has "
            f"{len(orig_node_ids)}; the graph cache is inconsistent"
        )

    centers = counter.centers_for(coords_nm)
    counts = counter.count_many(centers, root_id)
    vol_min, vol_max = counter.bounds

    return {
        "voxel_count": counts.astype(np.int32),
        "orig_node_ids": orig_node_ids,
        "clipped": clipped_flags(centers, vol_min, vol_max, counter.box_size),
        "center_vox": centers.astype(np.int32),
        "root_id": np.int64(root_id),
        "box_size": np.int32(counter.box_size),
        "mat_version": np.int32(counter.mat_version),
        "voxel_volume_nm3": np.int64(VOXEL_VOLUME_NM3),
        "resolution_nm": counter.resolution.astype(np.float64),
    }


def main() -> int:
    args = parse_args()
    rank, world = task_from_env()

    root_ids, n_nodes = load_cell_list(args.manifest)
    if args.limit:
        root_ids, n_nodes = root_ids[: args.limit], n_nodes[: args.limit]

    blocks = plan_shards(root_ids, n_nodes, world)
    mine = blocks[rank] if rank < len(blocks) else np.empty(0, np.int64)
    nodes_by_root = dict(zip(root_ids.tolist(), n_nodes.tolist()))

    args.output.mkdir(parents=True, exist_ok=True)

    def done(root_id: int) -> bool:
        return (args.output / f"{root_id}.npz").exists()

    pending = [int(r) for r in mine if args.redo or not done(int(r))]
    pending_nodes = sum(nodes_by_root[r] for r in pending)

    if args.dry_run:
        print(f"cells        {len(root_ids):,}  ({n_nodes.sum():,} nodes) across {world} ranks")
        print(f"this rank    {len(mine):,} cells, {sum(nodes_by_root[int(r)] for r in mine):,} nodes")
        print(f"pending      {len(pending):,} cells, {pending_nodes:,} nodes")
        print(f"output       {args.output}")
        print(f"\n  {'rank':>5} {'cells':>9} {'nodes':>13}")
        for index, block in enumerate(blocks):
            block_nodes = sum(nodes_by_root[int(r)] for r in block)
            marker = "  <- this rank" if index == rank else ""
            print(f"  {index:>5} {len(block):>9,} {block_nodes:>13,}{marker}")
        return 0

    counter = MaskVolumeCounter(
        mat_version=args.mat_version,
        box_size=args.box_size,
        num_threads=args.num_threads,
        cache_bytes=args.cache_bytes,
    )
    tqdm.write(
        f"rank {rank}/{world}: {len(pending):,} cells, {pending_nodes:,} nodes; "
        f"resolution {tuple(counter.resolution)} nm, box {counter.box_size}^3, "
        f"mat {counter.mat_version}"
    )

    n_done = n_nodes_done = n_failed = 0
    started = time.time()
    bar = tqdm(pending, desc=f"rank {rank}", unit="cell")
    for root_id in bar:
        graph_path = args.graph_cache / f"{root_id}.pt"
        if not graph_path.exists():
            tqdm.write(f"cell {root_id}: no graph cache file, skipped")
            n_failed += 1
            continue

        try:
            result = count_one_cell(counter, root_id, graph_path)
        except Exception as error:  # one bad cell must not end the shard
            tqdm.write(f"cell {root_id}: FAILED -- {type(error).__name__}: {error}")
            n_failed += 1
            continue

        # Write to a temporary name and rename, so a preempted rank cannot leave a
        # half-written npz that the resume logic would count as complete. The temp name
        # must itself end in .npz: savez_compressed appends the suffix when it is absent,
        # so a ".tmp" name silently becomes ".tmp.npz" and the rename finds nothing.
        tmp = args.output / f".{root_id}.tmp.npz"
        np.savez_compressed(tmp, **result)
        tmp.replace(args.output / f"{root_id}.npz")

        n_done += 1
        n_nodes_done += len(result["voxel_count"])
        elapsed = time.time() - started
        bar.set_postfix(
            nodes=f"{n_nodes_done:,}",
            rate=f"{n_nodes_done / max(elapsed, 1e-9):.0f}/s",
            failed=n_failed,
        )

    elapsed = time.time() - started
    tqdm.write(
        f"rank {rank}: {n_done:,} cells, {n_nodes_done:,} nodes in {elapsed / 60:.1f} min "
        f"({n_nodes_done / max(elapsed, 1e-9):.0f} nodes/s), {n_failed} failed"
    )
    return 1 if n_failed and not n_done else 0


if __name__ == "__main__":
    sys.exit(main())
