"""One-off check: does data/graph_cache/*.pt actually have a `pos` (xyz
coordinates, nm) attribute on the cached whole-cell Data objects? Cache
mtimes sit close to the commit that added `pos=` to
data/build_dataset_from_store.py, so this is worth confirming on real data
rather than assuming from source inspection alone -- see gnn/graph_transformer.py's
relative-position feature, which needs it. Also sanity-checks
data/geodesic_window.py::extract_window_subgraph's new rel_pos output on one
real window: center should land at (0,0,0), and the ordering (center at
local index 0) should hold.

Run via sbatch (mit_quicktest -- read-only, no training, no GPU needed):
    sbatch scripts/sbatch/check_pos_field.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.geodesic_window import extract_window_subgraph, window_membership  # noqa: E402

GRAPH_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "graph_cache"


def main() -> int:
    pt_files = sorted(GRAPH_CACHE_DIR.glob("*.pt"))[:3]
    if not pt_files:
        print(f"no .pt files found under {GRAPH_CACHE_DIR}")
        return 1

    for path in pt_files:
        data = torch.load(path, weights_only=False)
        print(f"\n{path.name}: x={tuple(data.x.shape)}", end="")
        if not hasattr(data, "pos") or data.pos is None:
            print("  -- NO `pos` ATTRIBUTE. rel_pos feature cannot be computed from this cache.")
            continue
        print(f"  pos={tuple(data.pos.shape)} dtype={data.pos.dtype}")
        # extract_window_subgraph also copies y_levels onto the window -- the
        # real pipeline (data/dataset_windowed.py) sets this from the
        # manifest before calling it; this standalone check doesn't load the
        # manifest, so a placeholder stands in (irrelevant to what's being
        # checked here: pos/rel_pos, not labels).
        data.y_levels = torch.zeros((1, 1), dtype=torch.long)

        mem_offsets, members = window_membership(
            data.edge_index.numpy(), data.edge_attr.numpy(), data.x.shape[0], 10_000.0
        )
        center = int(data.x.shape[0] // 2)  # arbitrary real node, not node 0
        window = extract_window_subgraph(data, center, mem_offsets, members)
        print(
            f"  window around node {center}: size={window.x.shape[0]}  "
            f"rel_pos[0] (should be ~0,0,0)={window.rel_pos[0].tolist()}  "
            f"rel_pos range=({window.rel_pos.min().item():.3f}, {window.rel_pos.max().item():.3f})"
        )
        assert np.allclose(window.rel_pos[0].numpy(), 0.0, atol=1e-5), "center node's rel_pos should be exactly 0"

    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
