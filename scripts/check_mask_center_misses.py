"""What are the ~0.5% of nodes whose center voxel does not hold their own root_id?

`scripts/check_mask_volume.py` samples center voxels and finds ~99.5% hold the cell. That
0.5% is small enough to be benign and large enough to be worth naming, because the
candidate explanations have very different implications:

* **Background (seg == 0)** -- the skeleton vertex sits in an unsegmented gap. Benign: a
  hole in the segmentation, not a coordinate error.
* **A different root_id** -- the vertex sits inside a *neighbouring* cell. Also expected
  at thin processes and branch points, where the skeleton centerline cuts a corner and
  passes momentarily through the neighbour, but worth quantifying: if this dominates and
  the offsets are large, it would point at a systematic shift instead.
* **Rounding** -- if `round()` centers hit while `trunc()` centers miss, the misses are an
  artifact of the truncation the pipeline deliberately inherits from training, not of the
  data. This is the one that would change how the numbers should be read, so it is tested
  directly rather than argued about.

Also reports, per miss, how far the nearest voxel of the cell actually is. A miss one voxel
from the cell is a centerline that grazes an edge; a miss tens of voxels away would be a
real coordinate problem.

Run via sbatch. Read-only, no writes to the cache.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.mask_volume import MaskVolumeCounter  # noqa: E402

CACHE_DIR = REPO / "data" / "mask_volume_cache"

#: Full-window reads are the expensive part, so cap how many misses get the detailed
#: nearest-voxel treatment. The classification above runs on every miss regardless.
MAX_DETAILED = 40


def main() -> int:
    files = sorted(CACHE_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"no npz in {CACHE_DIR}")

    counter = MaskVolumeCounter(mat_version=1718)
    seg = counter.crops.seg
    resolution = counter.resolution

    def voxel_at(point: np.ndarray) -> int:
        x, y, z = (int(v) for v in point)
        block = seg[x : x + 1, y : y + 1, z : z + 1].read().result()
        return int(np.asarray(block).ravel()[0])

    misses = []  # (root_id, node index, center, seg value, count, coords_nm)
    n_checked = 0

    for path in files:
        cell = np.load(path)
        root_id = int(cell["root_id"])
        centers = cell["center_vox"].astype(np.int64)
        counts = cell["voxel_count"]

        # Morton order so the chunk cache serves neighbouring reads, exactly as the
        # ingest does -- otherwise 25k single-voxel reads pull 25k separate chunks.
        from data.mask_volume import morton_order  # noqa: PLC0415

        for index in morton_order(centers):
            value = voxel_at(centers[index])
            n_checked += 1
            if value != root_id:
                misses.append((root_id, int(index), centers[index], value, int(counts[index])))

    print(f"cells    {len(files)}")
    print(f"checked  {n_checked:,} center voxels (every node, not a sample)")
    print(f"misses   {len(misses):,}  ({len(misses) / max(n_checked, 1):.3%})")
    if not misses:
        return 0

    kinds = Counter("background (seg == 0)" if value == 0 else "another root_id"
                    for _, _, _, value, _ in misses)
    print("\n=== what is actually at the center voxel ===")
    for kind, n in kinds.most_common():
        print(f"  {kind:<24} {n:>6,}  ({n / len(misses):.1%} of misses)")

    counts_at_miss = np.array([count for *_, count in misses])
    print("\n=== did the window still find the cell? ===")
    print(f"  nodes with count == 0     {int((counts_at_miss == 0).sum()):,}")
    print(f"  median count at a miss    {np.median(counts_at_miss):,.0f} voxels")
    print("  (a nonzero count means the cell is in the box, just not at the exact center)")

    print(f"\n=== nearest voxel of the cell, for up to {MAX_DETAILED} misses ===")
    box = counter.box_size
    offsets = []
    rounded_would_hit = 0
    for root_id, index, center, _value, _count in misses[:MAX_DETAILED]:
        mask = counter.crops.load_mask_block(center, root_id).numpy()[0]  # (Z, Y, X)
        if not mask.any():
            offsets.append(np.inf)
            continue
        # load_mask_block returns Z,Y,X; the center sits at box//2 on every axis either way.
        where = np.array(np.nonzero(mask)).T - (box // 2)
        # Voxel offsets are anisotropic (32/32/40 nm), so measure in nm to compare axes.
        nm = where * np.array([resolution[2], resolution[1], resolution[0]])
        offsets.append(float(np.linalg.norm(nm, axis=1).min()))

        # Would rounding instead of truncating have landed on the cell? The center is
        # trunc(coords/res), so the rounded center is at most one voxel away per axis.
        neighbourhood = mask[
            box // 2 - 1 : box // 2 + 2,
            box // 2 - 1 : box // 2 + 2,
            box // 2 - 1 : box // 2 + 2,
        ]
        rounded_would_hit += bool(neighbourhood.any())

    finite = np.array([o for o in offsets if np.isfinite(o)])
    if len(finite):
        print(f"  median distance to the cell   {np.median(finite):,.0f} nm")
        print(f"  p90 / max                     {np.percentile(finite, 90):,.0f} / "
              f"{finite.max():,.0f} nm")
        print(f"  (one voxel is 32 nm in x/y, 40 nm in z)")
    print(f"  cell present within +/-1 voxel of center: {rounded_would_hit}/"
          f"{len(offsets)}  -- i.e. a rounded center would have hit")

    return 0


if __name__ == "__main__":
    sys.exit(main())
