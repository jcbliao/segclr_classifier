"""Attributes per-window CPU extraction cost across window radii.

The training loop's throughput is set by data/geodesic_window.py::
extract_window_subgraph, which runs once per item in a DataLoader worker --
millions of times per epoch. When epoch time jumps at a larger radius, the
question is which part of that function grew, and the parts scale very
differently:

  - the Laplacian PE eigendecomposes a dense WxW matrix, so it is O(W^3) in
    the WINDOW size;
  - the edge remap masks over the whole cell's edge list, so it is O(E_cell)
    and does not grow with radius at all;
  - the two scratch arrays are allocated at whole-cell length, likewise
    independent of radius.

Timing them separately says whether a slow epoch is the radius or a fixed
per-item overhead that a bigger radius merely amortizes differently.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.geodesic_window import (  # noqa: E402
    DEFAULT_POS_DIM,
    _window_laplacian_pos_enc,
    extract_window_subgraph,
    membership_dir_name,
)

GRAPH_CACHE = Path(__file__).resolve().parent.parent / "data" / "graph_cache"


def main(args) -> int:
    torch.set_num_threads(1)  # a DataLoader worker gets one core; measure that
    paths = sorted(GRAPH_CACHE.glob("*.pt"))
    sample = [paths[i] for i in np.linspace(0, len(paths) - 1, args.n_cells).astype(int)]
    rng = np.random.default_rng(0)

    print(f"{'radius_um':>10}{'mean_W':>9}{'extract_us':>13}{'lpe_us':>10}{'lpe_share':>11}"
          f"{'edges/cell':>12}")
    for radius in args.radii:
        cache = Path(__file__).resolve().parent.parent / "data" / membership_dir_name(radius)
        if not cache.is_dir():
            print(f"{radius / 1000:>10.0f}  (no cache at {cache.name} -- skipped)")
            continue

        t_extract, t_lpe, n_win, w_total, e_total = 0.0, 0.0, 0, 0, 0
        for path in sample:
            data = torch.load(path, weights_only=False)
            data.y_levels = torch.zeros((1, 3), dtype=torch.long)
            npz = np.load(cache / f"{path.stem}.npz")
            mem_offsets, members = npz["mem_offsets"], npz["members"]
            n_nodes = data.x.shape[0]
            centers = rng.choice(n_nodes, size=min(args.n_windows, n_nodes), replace=False)
            e_total += data.edge_index.shape[1]

            for c in centers:
                t0 = time.perf_counter()
                w = extract_window_subgraph(data, int(c), mem_offsets, members, DEFAULT_POS_DIM)
                t_extract += time.perf_counter() - t0
                w_total += w.num_nodes
                n_win += 1

                # The same window's PE alone, to separate it from the rest.
                t0 = time.perf_counter()
                _window_laplacian_pos_enc(w.edge_index, w.num_nodes, DEFAULT_POS_DIM)
                t_lpe += time.perf_counter() - t0

        us_extract = t_extract / n_win * 1e6
        us_lpe = t_lpe / n_win * 1e6
        print(
            f"{radius / 1000:>10.0f}{w_total / n_win:>9.1f}{us_extract:>13.1f}{us_lpe:>10.1f}"
            f"{us_lpe / us_extract * 100:>10.0f}%{e_total / len(sample):>12.0f}"
        )

    print("\nlpe_us is measured on the extracted window, so it is also counted inside "
          "extract_us; lpe_share is its fraction of the whole extraction.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--radii", type=float, nargs="+", default=[10000.0, 20000.0, 40000.0])
    p.add_argument("--n-cells", type=int, default=5)
    p.add_argument("--n-windows", type=int, default=300)
    raise SystemExit(main(p.parse_args()))
