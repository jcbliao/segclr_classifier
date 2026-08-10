"""How much information is actually in `edge_attr` (skeleton edge length, nm)?

Nothing in gnn/ reads it today: GraphTransformer builds a binary adjacency via
to_dense_adj and ignores weights, and MPNNEncoder's SAGEConv ignores them too.
So it is computed, cached, sliced per window and batched to the GPU, then
dropped. Before wiring it in as an edge feature, the question is whether it
varies enough to carry signal -- CAVE skeletons are resampled to roughly
uniform vertex spacing, and a near-constant feature is a near-constant zero
contribution no matter how it is encoded.

Reports the spread of edge lengths overall and WITHIN windows (the scale the
model actually operates at), plus the per-window coefficient of variation.
CPU-only, reads the cached graphs. Run via sbatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.dataset_lcpn import load_manifest  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402

N_WINDOWS = 20000


def main() -> int:
    manifest = load_manifest()
    ds = WindowedGraphDatasetLCPN(manifest, "train")
    print(f"{len(ds)} windows over {len(ds.cell_data)} cells\n")

    all_len = np.concatenate(
        [d.edge_attr.numpy().reshape(-1) for d in list(ds.cell_data.values())[:200]]
    )
    print(f"=== whole-cell edge lengths, 200 cells ({len(all_len):,} edges) ===")
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  p{q:<3} {np.percentile(all_len, q):9.1f} nm")
    print(f"  mean {all_len.mean():9.1f}  std {all_len.std():9.1f}  "
          f"CV {all_len.std() / all_len.mean():.3f}")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(N_WINDOWS, len(ds)), replace=False)
    cvs, spans, n_edges = [], [], []
    for i in idx:
        w = ds[int(i)]
        e = w.edge_attr.numpy().reshape(-1)
        n_edges.append(len(e))
        if len(e) < 2:
            continue
        cvs.append(e.std() / max(e.mean(), 1e-9))
        spans.append(e.max() / max(e.min(), 1e-9))

    cvs, spans = np.array(cvs), np.array(spans)
    print(f"\n=== within-window variation, {len(idx)} windows ===")
    print(f"  edges per window: median {np.median(n_edges):.0f}, "
          f"{np.mean(np.array(n_edges) < 2):.1%} have <2 edges")
    print(f"  per-window CV of edge length:  median {np.median(cvs):.3f}  "
          f"p25 {np.percentile(cvs, 25):.3f}  p75 {np.percentile(cvs, 75):.3f}")
    print(f"  per-window max/min ratio:      median {np.median(spans):.2f}  "
          f"p90 {np.percentile(spans, 90):.2f}")
    print(
        "\n  A near-zero CV would mean edge length is effectively constant within a\n"
        "  window and cannot inform attention between its nodes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
