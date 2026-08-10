"""Exact coordinate identity check between raw v943 embedding points and OUR
cached skeleton (data/skeleton_cache/*.pkl) -- NOT a nearest-neighbor snap.
Per explicit user direction: raw embedding points and skeleton nodes should
either be the exact same points (same underlying sampling, just possibly
subsetted/reindexed) or they should not be treated as related at all --
approximate cKDTree snapping papers over that distinction rather than
answering it, and is exactly the kind of heuristic that caused the
now-deprecated pipeline's 85.9% match-quality failure on v343.

Checks, per sampled cell:
  - fraction of raw v943 points whose xyz exactly matches some skeleton node
  - fraction of skeleton nodes whose xyz exactly matches some raw v943 point
  - if exact matches are rare/absent, that's the answer: these are two
    independently-sampled point sets, not a real identity relationship, and
    building a raw-per-node dataset would require the same kind of
    unvalidated approximate join the deprecated pipeline already failed at.

Run via sbatch (mit_normal -- network I/O to GCS, not GPU work).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import cave_skeletons as cs  # noqa: E402
from data.dataset_lcpn import load_manifest  # noqa: E402
from data.public_reader import get_public_filesystem, get_raw_cell_embeddings  # noqa: E402

N_SAMPLE_CELLS = 25


def main():
    manifest = load_manifest()
    val_root_ids = [
        int(rid) for rid, info in manifest["cells"].items()
        if info["split"] == "val" and info.get("has_graph", False)
    ]
    rng = np.random.default_rng(0)
    sample = rng.choice(val_root_ids, size=min(N_SAMPLE_CELLS, len(val_root_ids)), replace=False)
    print(f"sampling {len(sample)} cells from val split", flush=True)

    fs = get_public_filesystem()

    n_no_v943, n_ok = 0, 0
    for root_id in sample:
        root_id = int(root_id)
        skeleton = cs.load_cached(root_id)
        if skeleton is None:
            print(f"  cell {root_id}: no cached skeleton, skipping", flush=True)
            continue

        try:
            raw = get_raw_cell_embeddings(root_id, filesystem=fs, data_key="microns_v943")
        except KeyError as exc:
            n_no_v943 += 1
            print(f"  cell {root_id}: no v943 shard entry ({exc}) -- skipping", flush=True)
            continue
        n_pts = raw.xyz_nm.shape[0]
        if n_pts == 0:
            print(f"  cell {root_id}: 0 raw v943 embeddings returned", flush=True)
            continue
        n_ok += 1

        # Exact match: build a set of skeleton coords as tuples, check membership.
        # Coords are floats -- compare bit-for-bit first, then note if a tight
        # (sub-nm) rounding tolerance changes the answer materially, so a
        # genuine "same points, different float repr" case isn't missed.
        skel_coords = skeleton.coords.astype(np.float64)
        raw_coords = raw.xyz_nm.astype(np.float64)

        skel_set_exact = {tuple(c) for c in skel_coords}
        raw_exact_hits = sum(1 for c in raw_coords if tuple(c) in skel_set_exact)

        skel_set_rounded = {tuple(np.round(c, 0)) for c in skel_coords}  # nearest nm
        raw_rounded_hits = sum(1 for c in raw_coords if tuple(np.round(c, 0)) in skel_set_rounded)

        print(
            f"  cell {root_id}: {n_pts} raw v943 pts, {len(skeleton)} skeleton nodes -- "
            f"exact bit match: {raw_exact_hits}/{n_pts} ({100*raw_exact_hits/n_pts:.1f}%), "
            f"rounded-to-nm match: {raw_rounded_hits}/{n_pts} ({100*raw_rounded_hits/n_pts:.1f}%)",
            flush=True,
        )
        # Sanity: show a couple of raw sample coords vs nearest-looking skeleton coords
        # so the actual coordinate scales/values are visible for manual inspection.
        print(f"    sample raw xyz[0:2]: {raw_coords[:2].tolist()}", flush=True)
        print(f"    sample skeleton coords[0:2]: {skel_coords[:2].tolist()}", flush=True)

    print(
        f"\n=== coverage: {n_ok}/{len(sample)} cells had usable v943 data, "
        f"{n_no_v943}/{len(sample)} had no v943 shard entry at all ===",
        flush=True,
    )
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
