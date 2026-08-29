"""De-risks the window-radius sweep before any cache is built or GPU spent.

Three things can go wrong when the radius stops being the 10um the pipeline
was written around, and all three are silent:

  1. window_nm=0 is supposed to give one single-node window per node -- the
     unaggregated case. If the bounded-Dijkstra kernel instead returned an
     empty membership, every window would be an empty graph and the readout
     would produce NaN rather than an error.
  2. A 1-node window has no edges at all, so the Laplacian PE and the
     attention mask run on a degenerate graph. An all-masked attention row
     comes out of softmax as NaN.
  3. Window size grows with radius, and the GraphTransformer's attention is
     quadratic in the padded node count -- so the 40um batch size has to be
     chosen from measured window sizes, not guessed.

Samples a few cells rather than scanning the cache: this answers "does the
shape of the thing change", which does not need all 2336.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from data.geodesic_window import extract_window_subgraph, window_membership  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402

GRAPH_CACHE = Path(__file__).resolve().parent.parent / "data" / "graph_cache"
RADII = [0.0, 10_000.0, 20_000.0, 40_000.0]
N_CELLS = 6

failures: list[str] = []

paths = sorted(GRAPH_CACHE.glob("*.pt"))
# Spread across the size distribution rather than taking the first N, which
# would all be whatever sorts first by root_id.
sample = [paths[i] for i in np.linspace(0, len(paths) - 1, N_CELLS).astype(int)]

print(f"=== window size vs radius ({N_CELLS} cells of {len(paths)}) ===")
print(f"{'radius_um':>10}{'cells':>7}{'nodes':>9}{'mean_W':>9}{'p50':>6}{'p95':>7}{'max_W':>7}{'entries':>12}")
per_radius: dict[float, dict] = {}
cells = [(p.stem, torch.load(p, weights_only=False)) for p in sample]

for radius in RADII:
    sizes_all = []
    total_entries = 0
    for _root_id, data in cells:
        n_nodes = data.x.shape[0]
        mem_offsets, members = window_membership(
            data.edge_index.numpy(), data.edge_attr.numpy(), n_nodes, radius
        )
        sizes = np.diff(mem_offsets)
        sizes_all.append(sizes)
        total_entries += len(members)
        if radius == 0.0:
            # The identity case must be exactly self, for every node.
            if not (sizes == 1).all():
                failures.append(
                    f"radius 0 on cell {_root_id}: window sizes not all 1 "
                    f"(min {sizes.min()}, max {sizes.max()})"
                )
            if not (members == np.arange(n_nodes)).all():
                failures.append(f"radius 0 on cell {_root_id}: members are not each node itself")
    sizes_all = np.concatenate(sizes_all)
    per_radius[radius] = {
        "mean": sizes_all.mean(), "p95": np.percentile(sizes_all, 95), "max": sizes_all.max(),
    }
    print(
        f"{radius / 1000:>10.0f}{len(cells):>7}{len(sizes_all):>9}{sizes_all.mean():>9.1f}"
        f"{np.percentile(sizes_all, 50):>6.0f}{np.percentile(sizes_all, 95):>7.0f}"
        f"{sizes_all.max():>7}{total_entries:>12}"
    )

# --- a real radius-0 batch through all three architectures ----------------
hierarchy = load_hierarchy(load_manifest())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=== radius-0 (single-node) windows through each architecture, on {device} ===")

_root_id, data = cells[0]
# Whole-cell Data straight from graph_cache has no y_levels -- the dataset
# attaches it per cell before extraction, and extract_window_subgraph carries
# it through to each window, so batching without it fails in collate.
data.y_levels = torch.zeros((1, hierarchy.depth), dtype=torch.long)
mem_offsets, members = window_membership(
    data.edge_index.numpy(), data.edge_attr.numpy(), data.x.shape[0], 0.0
)
windows = [extract_window_subgraph(data, c, mem_offsets, members) for c in range(min(64, data.x.shape[0]))]
print(f"window 0: n_nodes={windows[0].num_nodes} n_edges={windows[0].edge_index.shape[1]} "
      f"pos_enc={tuple(windows[0].pos_enc.shape)} rel_pos={tuple(windows[0].rel_pos.shape)}")
if windows[0].num_nodes != 1:
    failures.append(f"extract_window_subgraph at radius 0 gave {windows[0].num_nodes} nodes, expected 1")
if not torch.isfinite(windows[0].pos_enc).all():
    failures.append("radius-0 window pos_enc is not finite (Laplacian PE on a 1-node graph)")

batch = next(iter(DataLoader(windows, batch_size=32, shuffle=False))).to(device)
for name, cfg in (
    ("meanpool", ModelConfig(architecture="mean")),
    ("mpnn_L2_spatial", ModelConfig(architecture="mpnn", mpnn_layers=2, use_spatial_features=True)),
    ("gt_L4_H4", ModelConfig(architecture="graph_transformer", gt_depth=4, gt_heads=4)),
):
    try:
        model = WindowClassifier(cfg, hierarchy=hierarchy).to(device)
        g = model(batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc, rel_pos=batch.rel_pos)
        finite = bool(torch.isfinite(g).all())
        print(f"  {name:<18} g={tuple(g.shape)} all_finite={finite} mean={g.mean().item():+.4f}")
        if not finite:
            failures.append(f"{name}: non-finite readout on single-node windows")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  {name:<18} FAILED: {type(exc).__name__}: {exc}")

# --- what the 40um batch size should be ------------------------------------
print("\n=== attention footprint (GraphTransformer pads each batch to its max W) ===")
for radius in RADII:
    w = per_radius[radius]["max"]
    for bs in (4096, 2048, 1024):
        # (B, heads, W, W) fp32, one score matrix per layer, kept for backward.
        gib = 4096 * 0 + bs * 4 * w * w * 4 / 1024**3
        print(f"  radius {radius / 1000:>2.0f}um  max_W={w:>4}  batch={bs:>5}  "
              f"~{gib:>7.2f} GiB per attention matrix x depth 4")
        if gib < 2.0:
            break

print("\n=== result ===")
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    raise SystemExit(1)
print("all checks passed")
