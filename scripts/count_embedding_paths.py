"""How many centered paths does full route enumeration actually produce?

Enumerating every route through every node is combinatorial in the number of
branch points inside the budget. On a dendrite that is a handful of routes per
node; on a dense interneuron axon at 40 um it need not be. This counts, per
config, without allocating a single path, and extrapolates to the full cell set
so the storage decision is made on measured numbers.

    sbatch scripts/sbatch/count_embedding_paths.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.store_compat import use_v4  # noqa: E402

use_v4()

from data.embedding_paths import assert_forest, count_paths  # noqa: E402
from data.geodesic_window import build_csr_from_edges  # noqa: E402
from data.soma_restrict import (  # noqa: E402
    DEFAULT_SOMA_RADIUS_NM, nucleus_positions, restrict,
)

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
N_CELLS = 24

# Imported rather than restated, so the counter can never drift from what the
# builder actually writes.
from data.build_embedding_paths import CONFIGS as _CFG  # noqa: E402

CONFIGS = list(_CFG.items())


def main() -> int:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    cells = manifest["cells"]
    all_ids = [int(r) for r in cells]

    nuc = nucleus_positions(STORE_ROOT, root_ids=all_ids)
    have = [r for r in all_ids if r in nuc]
    print(f"cells: {len(all_ids)}   nucleus position present: {len(have)} "
          f"({100 * len(have) / len(all_ids):.1f}%)", flush=True)
    missing = [r for r in all_ids if r not in nuc]
    if missing:
        print(f"  missing ({len(missing)}): {missing[:6]}", flush=True)

    total_nodes_all = sum(c["n_nodes_covered"] for c in cells.values())
    print(f"covered nodes across all cells: {total_nodes_all:,}\n", flush=True)

    # A spread across the size range, not the head of the list.
    order = sorted(have, key=lambda r: cells[str(r)]["n_nodes_covered"])
    picks = [order[int(i * (len(order) - 1) / (N_CELLS - 1))] for i in range(N_CELLS)]

    sampled_nodes = 0
    agg = {tag: {"paths": 0, "max_node": 0, "arms_max": 0} for tag, _ in CONFIGS}
    rows = []

    for rid in picks:
        d = torch.load(ROOT / "data" / "graph_cache" / f"{rid}.pt", weights_only=False)
        r = restrict(d.pos.numpy(), d.edge_index.numpy(),
                     d.edge_attr.numpy().reshape(-1), nuc[rid], DEFAULT_SOMA_RADIUS_NM)
        offsets, neighbors, weights = r["csr"]
        n_kept = r["n_nodes_after"]
        if n_kept < 2:
            print(f"{rid}: {n_kept} nodes after restriction, skipped", flush=True)
            continue
        try:
            assert_forest(offsets, neighbors)
        except ValueError as exc:
            print(f"{rid}: {exc}", flush=True)
            continue

        sampled_nodes += n_kept
        ct = cells[str(rid)]["cell_type"]
        line = [f"{rid}  {ct:<10} {r['n_nodes_before']:>6,}->{n_kept:>6,} nodes "
                f"{r['n_components']:>4} comp"]
        for tag, kw in CONFIGS:
            per_node, arms = count_paths(offsets, neighbors, weights, **kw)
            agg[tag]["paths"] += int(per_node.sum())
            agg[tag]["max_node"] = max(agg[tag]["max_node"], int(per_node.max()))
            agg[tag]["arms_max"] = max(agg[tag]["arms_max"], int(arms.max()))
            line.append(f"    {tag:>7}: {per_node.sum():>12,} paths  "
                        f"median {np.median(per_node):>6.1f}  p99 "
                        f"{np.percentile(per_node, 99):>9,.0f}  max {per_node.max():>10,}")
        print("\n".join(line) + "\n", flush=True)
        rows.append(rid)

    scale = total_nodes_all / max(sampled_nodes, 1)
    print("=" * 78, flush=True)
    print(f"sampled {len(rows)} cells / {sampled_nodes:,} restricted nodes; "
          f"extrapolation factor x{scale:.1f} to all {total_nodes_all:,} covered nodes\n",
          flush=True)
    print(f"{'config':>8}  {'paths (sampled)':>18}  {'paths (projected)':>20}  "
          f"{'max/node':>12}  {'est. size':>12}", flush=True)
    for tag, _ in CONFIGS:
        p = agg[tag]["paths"]
        proj = p * scale
        # 4 bytes per node id; mean path length approximated by the node budget
        approx_len = {"10node": 11, "20node": 21, "40node": 41}.get(tag, 0)
        size = f"{proj * approx_len * 4 / 1e9:,.1f} GB" if approx_len else "see nodes/path"
        print(f"{tag:>8}  {p:>18,}  {proj:>20,.0f}  {agg[tag]['max_node']:>12,}  {size:>12}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
