"""Measures peak GraphTransformer VRAM per (window radius, batch size), to
decide whether the GT runs can move off H200s onto the far more numerous
L40S pool.

The GraphTransformer pads each batch to the LARGEST window in that batch
(to_dense_batch / to_dense_adj) and its attention is quadratic in that padded
width, so cost is driven by the batch maximum, not the mean. With thousands
of windows drawn at random per batch, the batch max sits near the global max
essentially every step -- so the honest probe is a deliberately worst-case
batch built from the largest windows in the split, not a random one. Both are
measured here; the worst case is the one that decides the answer.

Reports torch.cuda.max_memory_allocated() around a real forward + backward,
which is what actually has to fit, and compares it against both cards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402

L40S_GIB = 48.0
H200_GIB = 141.0


def peak_gib(model, windows, batch_size, device) -> tuple[float, int]:
    """Peak allocated GiB over one forward+backward on these windows, plus
    the padded width the batch actually reached."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(DataLoader(windows[:batch_size], batch_size=batch_size, shuffle=False)))
    batch = batch.to(device)
    padded_w = int(torch.bincount(batch.batch).max().item())
    g = model(batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
    loss = model.cls_head.compute_loss(g, batch.y_levels)
    loss.backward()
    model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    del batch, g, loss
    torch.cuda.empty_cache()
    return peak, padded_w


def main(args) -> int:
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)} "
          f"({torch.cuda.get_device_properties(device).total_memory / 1024**3:.0f} GiB)")
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)

    for radius in args.radii:
        # The test split is the smaller one and holds the same size
        # distribution; loading train too would double the host RAM for no
        # extra information about window widths.
        ds = WindowedGraphDatasetLCPN(manifest, "test", window_nm=radius)
        sizes = np.array([
            int(np.diff(ds.cell_membership[r][0])[c])
            for r, c in zip(ds.index_root_ids, ds.index_centers, strict=True)
        ])
        order = np.argsort(-sizes)
        print(f"\n=== radius {radius / 1000:g}um: {len(ds)} windows, "
              f"mean W {sizes.mean():.1f}, p99 {np.percentile(sizes, 99):.0f}, max {sizes.max()} ===")

        model = WindowClassifier(
            ModelConfig(
                architecture="graph_transformer", gt_depth=4, gt_heads=4,
                gt_dim=args.gt_dim, use_embeddings=not args.no_embeddings,
            ),
            hierarchy,
        ).to(device)

        for batch_size in args.batch_sizes:
            for label, idx in (("worst-case", order), ("random", np.random.default_rng(0).permutation(len(ds)))):
                windows = [ds[int(i)] for i in idx[:batch_size]]
                try:
                    gib, padded_w = peak_gib(model, windows, batch_size, device)
                    verdict = (
                        "L40S ok" if gib < L40S_GIB * 0.85
                        else ("H200 only" if gib < H200_GIB * 0.85 else "TOO BIG")
                    )
                    print(f"  batch={batch_size:>5} {label:<11} padded_W={padded_w:>4}  "
                          f"peak={gib:>7.2f} GiB   {verdict}")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  batch={batch_size:>5} {label:<11} OOM on this card")
        del model
        torch.cuda.empty_cache()

    print(f"\nthresholds: L40S {L40S_GIB:.0f} GiB, H200 {H200_GIB:.0f} GiB "
          f"(verdict uses an 85% headroom margin for allocator fragmentation)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--radii", type=float, nargs="+", default=[20000.0, 40000.0])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4096, 2048, 1024])
    p.add_argument("--gt-dim", type=int, default=128)
    p.add_argument("--no-embeddings", action="store_true")
    raise SystemExit(main(p.parse_args()))
