"""Rigorous check of the embedding-to-skeleton-node nearest-neighbor match
quality, across the real built dataset -- not the single-cell anecdote from
scripts/explore_cave_alignment.py, and not compared against cell span (the
wrong yardstick: a >1mm cell doesn't tell you whether a 740nm residual is
large or small). The right comparison is against LOCAL scale: how does the
match residual compare to the skeleton's own per-node radius at that point?
A residual much larger than the local process radius means the matched
embedding may have landed on a neighboring branch (or even a different
neurite) rather than genuinely on the same piece of skeleton.

Recomputes residuals directly from the cached skeletons + public embeddings
(does not rely on anything build_dataset.py stored, since it only kept the
median). Run via sbatch (mit_normal -- CPU, no GPU needed, just numpy/scipy
over already-cached data + one already-authenticated GCS read pass).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from data import cave_skeletons as cs  # noqa: E402
from data import public_reader as pr  # noqa: E402
from data.dataset import load_manifest  # noqa: E402


def main() -> int:
    manifest = load_manifest()
    root_ids = sorted(int(r) for r in manifest["cells"])
    print(f"{len(root_ids)} cells in manifest")

    fs = pr.get_public_filesystem()

    n_no_radius = 0
    all_residuals = []
    all_ratios = []  # residual / local radius, where radius is available
    per_cell_summary = []

    for root_id in root_ids:
        skeleton = cs.load_cached(root_id)
        if skeleton is None:
            continue
        cell = pr.get_raw_cell_embeddings(
            root_id, fs, data_key="microns_nm_coord_public_offset_v343"
        )
        if cell.embeddings.shape[0] == 0:
            continue

        tree = cKDTree(skeleton.coords.astype(np.float64))
        dist, node_idx = tree.query(cell.xyz_nm.astype(np.float64))
        all_residuals.append(dist)

        if skeleton.radii is not None:
            local_radius = skeleton.radii[node_idx].astype(np.float64)
            valid = local_radius > 0
            if valid.any():
                ratio = dist[valid] / local_radius[valid]
                all_ratios.append(ratio)
                per_cell_summary.append(
                    {
                        "root_id": root_id,
                        "median_residual_nm": float(np.median(dist)),
                        "median_radius_nm": float(np.median(local_radius[valid])),
                        "median_ratio": float(np.median(ratio)),
                        "frac_ratio_gt_1": float((ratio > 1).mean()),
                        "frac_ratio_gt_2": float((ratio > 2).mean()),
                    }
                )
        else:
            n_no_radius += 1

    if not all_residuals:
        print("no cells with both a cached skeleton and embeddings -- nothing to check")
        return 1

    residuals = np.concatenate(all_residuals)
    print("\n=== residual distance (embedding xyz -> nearest CAVE skeleton node), across ALL cells ===")
    print(
        f"n={len(residuals)}  median={np.median(residuals):.1f}nm  mean={residuals.mean():.1f}nm  "
        f"p90={np.percentile(residuals, 90):.1f}nm  p99={np.percentile(residuals, 99):.1f}nm  "
        f"max={residuals.max():.1f}nm"
    )

    print(f"\ncells with no radius field on their skeleton: {n_no_radius}/{len(root_ids)} "
          f"(radius is optional -- 'older skeletons lack them' per segclr_db.skeletons)")

    if all_ratios:
        ratios = np.concatenate(all_ratios)
        print("\n=== residual / local process radius, where radius is available ===")
        print(
            f"n={len(ratios)}  median={np.median(ratios):.2f}  mean={ratios.mean():.2f}  "
            f"p90={np.percentile(ratios, 90):.2f}  max={ratios.max():.2f}"
        )
        print(f"fraction with residual > 1x local radius (match landed outside the process): "
              f"{(ratios > 1).mean():.1%}")
        print(f"fraction with residual > 2x local radius (likely wrong branch/neurite): "
              f"{(ratios > 2).mean():.1%}")

        per_cell_summary.sort(key=lambda d: -d["median_ratio"])
        print("\nworst 10 cells by median residual/radius ratio:")
        for d in per_cell_summary[:10]:
            print(f"  {d['root_id']}: median_ratio={d['median_ratio']:.2f}  "
                  f"median_residual={d['median_residual_nm']:.0f}nm  "
                  f"median_radius={d['median_radius_nm']:.0f}nm  "
                  f"frac>1x={d['frac_ratio_gt_1']:.1%}  frac>2x={d['frac_ratio_gt_2']:.1%}")
    else:
        print("\nno cells had a usable radius field -- cannot compute residual/radius ratio. "
              "Falling back to residual-distance-only interpretation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
