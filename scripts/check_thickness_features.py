"""Validates the dendrite-thickness node feature against REAL cached data --
the id-based join, the window slice, and how much of the feature is actually
measured rather than masked out.

The synthetic smoke test (scripts/smoke_test_model.py) can only prove the
model consumes a (N, 2) tensor of the right width. It cannot catch a bad join:
the thickness cache is indexed by SKELETON vertex while the graph cache holds
only the embedded subset, and `orig_node_ids` is what bridges them. A
silently-wrong index there would still produce correctly-shaped garbage.

Also reports the measured fraction per class. That number decides how to read
any result from --gt-use-thickness: the cache is NaN for every axon node,
every branch point, and every mesh-hole miss, so if measurement coverage
itself correlates with cell type, the model can score better by reading the
mask channel as a class hint rather than the radius as biology. Better to know
that before running the comparison than to explain a suspiciously good number
afterwards.

Run via sbatch (mit_normal, CPU-only -- no model, no GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.dataset_lcpn import load_manifest  # noqa: E402
from data.dataset_windowed import (  # noqa: E402
    THICKNESS_CACHE_DIR_NAME,
    WindowedGraphDatasetLCPN,
    load_thickness_features,
)
from data.geodesic_window import THICKNESS_SCALE_NM  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
N_CELLS_DIRECT = 5   # cells to check the raw join on, against the npz itself
N_WINDOWS = 2000     # windows to pull through the real dataset path


def main() -> int:
    manifest = load_manifest()
    graph_cache = REPO_ROOT / "data" / "graph_cache"
    thickness_dir = REPO_ROOT / "data" / THICKNESS_CACHE_DIR_NAME

    root_ids = sorted(int(r) for r in manifest["cells"])
    n_cached = len(list(thickness_dir.glob("*.npz")))
    print(f"{len(root_ids)} cells in manifest, {n_cached} thickness cache files\n")

    missing = [r for r in root_ids if not (thickness_dir / f"{r}.npz").exists()]
    print(f"cells with no thickness cache: {len(missing)}" + (f" -> {missing[:5]}" if missing else ""))

    # --- 1. the join itself, against the npz directly -----------------------
    print(f"\n=== raw join, {N_CELLS_DIRECT} cells ===")
    for root_id in root_ids[:N_CELLS_DIRECT]:
        data = torch.load(graph_cache / f"{root_id}.pt", weights_only=False)
        n_nodes = data.x.shape[0]
        raw = np.load(thickness_dir / f"{root_id}.npz")["radius_nm"]
        oid = data.orig_node_ids.numpy()

        # The join is only meaningful if every graph node indexes a real
        # skeleton vertex -- an off-by-one or a stale cache shows up here.
        assert oid.max() < len(raw), (
            f"{root_id}: orig_node_ids max {oid.max()} >= skeleton length {len(raw)} "
            "-- graph cache and thickness cache disagree about this cell"
        )

        feat, found = load_thickness_features(thickness_dir / f"{root_id}.npz", data.orig_node_ids, n_nodes)
        assert found and feat.shape == (n_nodes, 2), (found, feat.shape)
        assert torch.isfinite(feat).all(), f"{root_id}: non-finite value survived the NaN mask"

        expected = raw[oid]
        measured = np.isfinite(expected)
        assert np.array_equal(feat[:, 1].numpy().astype(bool), measured), (
            f"{root_id}: measured flag disagrees with the cache's own NaN pattern"
        )
        # Channel 0 must be the actual radius where measured, and exactly 0
        # where not -- not a stale value left behind by the masking.
        assert np.allclose(
            feat[measured, 0].numpy(), expected[measured] / THICKNESS_SCALE_NM, atol=1e-6
        ), f"{root_id}: radius channel does not match the cache"
        assert (feat[~torch.from_numpy(measured), 0] == 0).all(), f"{root_id}: unmeasured radius not zeroed"

        r = expected[measured]
        print(
            f"  {root_id}: {n_nodes:6d} nodes, {measured.mean():5.1%} measured, "
            f"radius nm median={np.median(r):7.1f} p5={np.percentile(r, 5):7.1f} "
            f"p95={np.percentile(r, 95):7.1f}" if measured.any()
            else f"  {root_id}: {n_nodes:6d} nodes, 0% measured"
        )

    # --- 2. through the real dataset path -----------------------------------
    print(f"\n=== dataset path (train split, {N_WINDOWS} windows) ===")
    ds = WindowedGraphDatasetLCPN(manifest, "train", use_thickness=True)
    print(f"  {len(ds)} windows over {len(ds.cell_data)} cells")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(N_WINDOWS, len(ds)), replace=False)
    n_nodes_seen = 0
    n_measured = 0
    for i in idx:
        w = ds[int(i)]
        assert hasattr(w, "thickness"), "window has no thickness attribute"
        assert w.thickness.shape == (w.x.shape[0], 2), (w.thickness.shape, w.x.shape)
        assert torch.isfinite(w.thickness).all(), "non-finite thickness in a window"
        n_nodes_seen += w.x.shape[0]
        n_measured += int(w.thickness[:, 1].sum())
    print(f"  {n_nodes_seen} window-nodes, {n_measured / max(n_nodes_seen, 1):.1%} measured")

    # --- 3. measured fraction by class --------------------------------------
    # See the module docstring: if coverage tracks cell type, the mask channel
    # is a label hint and any gain from this feature is partly spurious.
    print("\n=== measured fraction by cell type (whole train split) ===")
    by_class: dict[str, list[float]] = {}
    for root_id, info in manifest["cells"].items():
        if info["split"] != "train":
            continue
        rid = int(root_id)
        if rid not in ds.cell_data:
            continue
        t = ds.cell_data[rid].thickness
        by_class.setdefault(info["cell_type"], []).append(float(t[:, 1].mean()))

    print(f"  {'class':<16} {'n_cells':>8} {'mean measured':>14}")
    fracs = []
    for cls in sorted(by_class):
        vals = by_class[cls]
        fracs.append(float(np.mean(vals)))
        print(f"  {cls:<16} {len(vals):>8} {np.mean(vals):>13.1%}")
    if fracs:
        print(
            f"\n  spread across classes: min={min(fracs):.1%} max={max(fracs):.1%} "
            f"(a wide spread means the measured flag leaks class information)"
        )

    print("\nall thickness feature checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
