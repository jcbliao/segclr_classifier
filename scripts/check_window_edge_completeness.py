"""Verifies data/geodesic_window.py::extract_window_subgraph includes EVERY
edge between window-member nodes (the full induced subgraph, not a
subsample) and reports real edge/node ratios -- a direct empirical check,
not just a code-reading argument, for whether a ~10-node window's edge_index
is missing anything.

For a sample of cells: for each window, independently recomputes the
induced-edge count by brute-force filtering the cell's FULL edge_index
against the window's node set (a second, simpler implementation of the same
filter extract_window_subgraph does) and checks it's IDENTICAL to what
extract_window_subgraph itself produces -- an actual correctness check, not
just re-reading the same code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from data.dataset_lcpn import load_hierarchy, load_manifest
from data.geodesic_window import extract_window_subgraph, window_membership

GRAPH_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "graph_cache"
N_CELLS = 30
WINDOW_NM = 10_000.0


def main() -> int:
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    test_cells = [rid for rid, info in manifest["cells"].items() if info["split"] == "test"][:N_CELLS]

    total_windows, total_nodes, total_edges_undirected = 0, 0, 0
    mismatches = 0

    for rid in test_cells:
        data = torch.load(GRAPH_CACHE_DIR / f"{rid}.pt", weights_only=False)
        n = data.x.shape[0]
        # extract_window_subgraph copies data.y_levels onto every window it
        # cuts -- the cached .pt predates the LCPN pivot and has none, same
        # as data/dataset_windowed.py::WindowedGraphDatasetLCPN.__init__
        # attaches it at load time, so this has to too.
        cell_type = manifest["cells"][rid]["cell_type"]
        path = hierarchy.label_paths[cell_type]
        levels = [hierarchy.level_maps[lvl][path[lvl]] for lvl in range(hierarchy.depth)]
        data.y_levels = torch.tensor(levels, dtype=torch.long).unsqueeze(0)
        mem_offsets, members = window_membership(
            data.edge_index.numpy(), data.edge_attr.numpy(), n, WINDOW_NM
        )
        edge_index_np = data.edge_index.numpy()

        for center in range(n):
            window = extract_window_subgraph(data, center, mem_offsets, members)
            w_nodes = members[mem_offsets[center]: mem_offsets[center + 1]]

            # Independent re-derivation of the induced edge count, straight
            # from the cell's full edge list -- not reusing
            # extract_window_subgraph's own filtering logic.
            in_w = np.zeros(n, dtype=bool)
            in_w[w_nodes] = True
            keep = in_w[edge_index_np[0]] & in_w[edge_index_np[1]]
            expected_directed_edges = int(keep.sum())

            actual_directed_edges = window.edge_index.shape[1]
            if actual_directed_edges != expected_directed_edges:
                mismatches += 1

            total_windows += 1
            total_nodes += len(w_nodes)
            total_edges_undirected += actual_directed_edges // 2  # stored both directions

    print(f"checked {total_windows} windows across {len(test_cells)} cells")
    print(f"mismatches between extract_window_subgraph and independent recomputation: {mismatches}")
    print(f"avg nodes/window: {total_nodes / total_windows:.2f}")
    print(f"avg UNDIRECTED edges/window: {total_edges_undirected / total_windows:.2f}")
    print(f"avg edges per node (undirected): {total_edges_undirected / total_nodes:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
