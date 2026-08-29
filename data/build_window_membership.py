"""Precomputes per-cell geodesic-window membership (data/geodesic_window.py)
for every cell in data/graph_cache/*.pt (segclr_db-store raw 64-dim
embeddings + skeleton edges) -- a one-time offline cost, not something
recomputed per training step (see that module's docstring for why this is
safe: membership depends only on skeleton structure + window_nm, never on
masking randomness or which embedding dim is attached, so it's stable across
all epochs of training and independent of the embedding source).

The default window_nm=10000 (10 microns) matches the baseline's own
aggregation window (see CLAUDE.md's project-goal section) -- that is what
makes the GNN's per-window classification comparable to the baseline's.
--window-nm sweeps it deliberately; window_nm=0 is the identity case, one
window per node containing only that node, which is how the baseline's own
aggregate.py expresses "raw, unaggregated".

Writes {out_dir}/{root_id}.npz (mem_offsets, members, both int32 -- cells max
out around 20k nodes per CLAUDE.md's p99, so int32 has enormous headroom over
int64 while halving storage), where out_dir is
data/geodesic_window.py::membership_dir_name(window_nm) -- one directory per
radius, so radii can't serve each other's windows.

Run via sbatch (mit_normal, no GPU needed -- this is pure CPU/numba):
    python data/build_window_membership.py                    # 10um
    python data/build_window_membership.py --window-nm 40000  # 40um
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.geodesic_window import (  # noqa: E402
    DEFAULT_WINDOW_NM,
    membership_dir_name,
    window_membership,
)

GRAPH_CACHE_DIR = Path(__file__).resolve().parent / "graph_cache"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger("build_window_membership")


def main(args) -> int:
    window_nm = float(args.window_nm)
    out_dir = Path(__file__).resolve().parent / membership_dir_name(window_nm)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("window_nm=%.0f -> %s", window_nm, out_dir)
    cell_paths = sorted(GRAPH_CACHE_DIR.glob("*.pt"))
    todo = [p for p in cell_paths if not (out_dir / f"{p.stem}.npz").exists()]
    logger.info("%d cells total, %d already done, %d to do", len(cell_paths), len(cell_paths) - len(todo), len(todo))

    total_entries, total_nodes = 0, 0
    t0 = time.monotonic()
    for path in tqdm(todo, desc="cells", unit="cell"):
        data = torch.load(path, weights_only=False)
        n_nodes = data.x.shape[0]
        mem_offsets, members = window_membership(
            data.edge_index.numpy(), data.edge_attr.numpy(), n_nodes, window_nm
        )
        total_entries += len(members)
        total_nodes += n_nodes
        np.savez(
            out_dir / f"{path.stem}.npz",
            mem_offsets=mem_offsets.astype(np.int32),
            members=members.astype(np.int32),
        )

    elapsed = time.monotonic() - t0
    if todo:
        logger.info(
            "built %d cells in %.0fs (%.2f cells/s), avg window size %.1f nodes, %d total membership entries",
            len(todo), elapsed, len(todo) / max(elapsed, 1e-9),
            total_entries / max(total_nodes, 1), total_entries,
        )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--window-nm", type=float, default=DEFAULT_WINDOW_NM,
        help="geodesic window RADIUS in nm (default 10000 = the baseline's 10um window). "
             "0 gives one single-node window per node -- the unaggregated case.",
    )
    raise SystemExit(main(p.parse_args()))
