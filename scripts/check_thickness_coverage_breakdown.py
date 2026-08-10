"""Decomposes "unmeasured" in the dendrite-thickness cache into its actual
causes, because the single measured/unmeasured number conflates two very
different things.

data/build_dendrite_thickness.py measures a vertex only if it is
`finite tangent & dendrite compartment & degree <= 2`. So a NaN in the cache
means one of:

  1. not a dendrite compartment (axon or soma)   -- by design, never eligible
  2. a branch point (degree > 2)                 -- by design, never eligible
  3. a degenerate local tangent                  -- by design, never eligible
  4. eligible, but ray casting returned nothing  -- an actual FAILURE
     (mesh hole, empty patch, all rays missing)

Only (4) is a measurement failure. (1)-(3) are the estimator refusing to
report a shaft radius where the concept does not apply. Reading a low measured
fraction as "the method mostly failed" is wrong if it is dominated by (1).

This matters for interpretation, not just bookkeeping: the measured fraction
correlates strongly with cell type (scripts/check_thickness_features.py:
17.8% PV to 59.1% L5ET). If that spread is driven by (1), it is a statement
about each cell type's axon/dendrite composition in the skeleton -- real
biology, though still a label leak for the model. If it is driven by (4), it
is a mesh-quality artifact. Those call for different conclusions.

Eligibility is recomputed here from the cached SKELETON alone, exactly as the
builder computes it -- no mesh fetch, no CAVE call.

Run via sbatch (mit_normal, CPU-only).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import cave_skeletons as cs  # noqa: E402
from data import dendrite_thickness as dt  # noqa: E402
from data.dataset_lcpn import load_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_CACHE = REPO_ROOT / "data" / "graph_cache"
THICKNESS_CACHE = REPO_ROOT / "data" / "dendrite_thickness_cache"

CATEGORIES = ["measured", "failed", "branch_point", "bad_tangent", "axon", "soma", "other_compartment"]


def classify_one_cell(root_id: int) -> tuple[np.ndarray, int] | None:
    """Per-graph-node category counts for one cell, restricted to the nodes the
    model actually sees (i.e. indexed through orig_node_ids, not all skeleton
    vertices) -- those are the nodes whose feature channel is at stake."""
    npz_path = THICKNESS_CACHE / f"{root_id}.npz"
    if not npz_path.exists():
        return None

    with open(cs._cache_path(root_id), "rb") as f:
        skel = pickle.load(f)

    data = torch.load(GRAPH_CACHE / f"{root_id}.pt", weights_only=False)
    oid = data.orig_node_ids.numpy()

    rc_skel = dt.skeleton_for_ray_casting(skel)
    tangents, _ = dt.local_tangents(rc_skel)
    _, _, degree = dt.skeleton_neighbors(rc_skel)

    radius = np.load(npz_path)["radius_nm"][oid]
    comp = np.asarray(skel.compartments)[oid]
    good_tangent = np.isfinite(tangents).all(axis=1)[oid]
    deg = degree[oid]

    is_dendrite = np.isin(comp, dt.DENDRITE_COMPARTMENTS)
    measured = np.isfinite(radius)
    eligible = good_tangent & is_dendrite & (deg <= 2)

    counts = np.zeros(len(CATEGORIES), dtype=np.int64)
    counts[0] = measured.sum()
    # Precedence below mirrors the builder's own conjunction: compartment
    # first (the coarsest reason a vertex is out of scope), then branch point,
    # then tangent. A node can satisfy several; it is attributed once, to the
    # broadest applicable reason, so the columns sum to the node count.
    rest = ~measured
    counts[1] = (rest & eligible).sum()                                    # failed
    counts[2] = (rest & ~eligible & is_dendrite & (deg > 2)).sum()         # branch point
    counts[3] = (rest & ~eligible & is_dendrite & (deg <= 2) & ~good_tangent).sum()
    counts[4] = (rest & ~eligible & (comp == dt.COMPARTMENT_AXON)).sum()
    counts[5] = (rest & ~eligible & (comp == dt.COMPARTMENT_SOMA)).sum()
    counts[6] = rest.sum() - counts[1:6].sum()
    return counts, len(oid)


def main(args) -> int:
    manifest = load_manifest()
    cells = [(int(r), i) for r, i in manifest["cells"].items()]
    if args.limit:
        cells = cells[: args.limit]

    total = np.zeros(len(CATEGORIES), dtype=np.int64)
    by_class: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(CATEGORIES), dtype=np.int64))
    n_done = n_skipped = 0

    for root_id, info in cells:
        try:
            res = classify_one_cell(root_id)
        except FileNotFoundError:
            res = None
        if res is None:
            n_skipped += 1
            continue
        counts, _ = res
        total += counts
        by_class[info["cell_type"]] += counts
        n_done += 1

    print(f"classified {n_done} cells ({n_skipped} skipped for a missing skeleton/thickness cache)\n")

    n = total.sum()
    print(f"=== all nodes ({n:,}) ===")
    for name, c in zip(CATEGORIES, total, strict=True):
        print(f"  {name:<20} {c:>12,}  {c / n:6.1%}")

    eligible_total = total[0] + total[1]
    print(
        f"\n  of the {eligible_total:,} nodes that were ELIGIBLE (dendrite, non-branch, good "
        f"tangent),\n  {total[0] / max(eligible_total, 1):.1%} were measured successfully -- "
        f"ray-cast failure rate {total[1] / max(eligible_total, 1):.2%}"
    )
    print(
        f"  the other {n - eligible_total:,} nodes ({(n - eligible_total) / n:.1%}) were never "
        f"eligible by design, not failures"
    )

    print("\n=== by cell type: what the unmeasured nodes actually are ===")
    hdr = f"  {'class':<14} {'nodes':>10} {'measured':>9} {'axon':>8} {'branch':>8} {'soma':>7} {'failed':>8}"
    print(hdr)
    for cls in sorted(by_class, key=lambda c: -by_class[c][0] / max(by_class[c].sum(), 1)):
        c = by_class[cls]
        t = max(c.sum(), 1)
        print(
            f"  {cls:<14} {c.sum():>10,} {c[0] / t:>8.1%} {c[4] / t:>7.1%} {c[2] / t:>7.1%} "
            f"{c[5] / t:>6.1%} {c[1] / t:>7.1%}"
        )

    print(
        "\nRead the 'axon' column against 'measured': where they trade off, the coverage\n"
        "spread across cell types is skeleton composition (how much of each cell type's\n"
        "reconstructed arbor is axon vs. dendrite), not the estimator failing."
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="only the first N cells (0 = all)")
    main(p.parse_args())
