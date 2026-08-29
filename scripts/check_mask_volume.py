"""Validate data/mask_volume_cache: are these counts real, and are they the right cell's?

The failure this exists to catch is silent. Masking a root_id against the wrong
materialization's segmentation, or an off-by-one in the window geometry, does not raise --
it returns counts that look like numbers. So every check here is one that a wrong-volume
run would fail and a correct one cannot.

1. **The center voxel is the cell.** A skeleton vertex sits on the cell's centerline, so
   the voxel at the window's center must belong to that root_id. This is re-read from the
   segmentation rather than taken from the cache, and it is the single strongest check: a
   1300 id against the 1718 volume finds nothing, and this goes to ~0% immediately.
2. **Counts are in range** -- never negative, never above box_size^3, and nonzero for
   effectively every node.
3. **Occupancy is physically plausible.** A neurite through a 4128 nm box fills order 1%
   of it, not 50% and not 0.001%.
4. **Volume tracks measured thickness.** On nodes where the ray-cast dendrite radius was
   measured, count should rise with radius^2 -- an independent estimate of the same
   physical quantity, from a different pipeline (mesh ray casting) and a different data
   source. A positive Spearman correlation here is hard to get by accident.

Run via sbatch (mit_quicktest is enough). Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.mask_volume import MaskVolumeCounter  # noqa: E402

CACHE_DIR = REPO / "data" / "mask_volume_cache"
THICKNESS_DIR = REPO / "data" / "dendrite_thickness_cache"

#: Voxels per window at the default box size, for occupancy percentages.
def _box_voxels(box_size: int) -> int:
    return int(box_size) ** 3


def load_cache(cache_dir: Path) -> list[dict]:
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no npz files in {cache_dir}; run data/build_mask_volume.py first")
    return [dict(np.load(path)) | {"path": path} for path in files]


def check_center_voxel(cells: list[dict], n_sample: int = 200,
                       max_cells: int = 40) -> None:
    """Re-read the segmentation at each sampled window's center voxel.

    Capped at ``max_cells`` because each single-voxel read pulls (and gzip-
    decompresses) a whole 256x256x32 chunk, and these reads are serial: 200
    samples across all 2335 cells would be ~467k chunk pulls and run for hours to
    answer a question that 40 cells settle. The cells are sampled at random
    rather than taken in order, so the subset is not all one region of the volume.
    """
    print("\n=== 1. center voxel belongs to the cell (re-read from the volume) ===")
    rng = np.random.default_rng(0)
    counter = MaskVolumeCounter(mat_version=int(cells[0]["mat_version"]))
    seg = counter.crops.seg

    if len(cells) > max_cells:
        picked = rng.choice(len(cells), size=max_cells, replace=False)
        cells = [cells[i] for i in picked]
        print(f"  (sampling {max_cells} cells of the cache; each read pulls a whole chunk)")

    n_hit = n_checked = 0
    for cell in cells:
        centers = cell["center_vox"].astype(np.int64)
        root_id = int(cell["root_id"])
        if not len(centers):
            continue
        take = rng.choice(len(centers), size=min(n_sample, len(centers)), replace=False)
        for index in take:
            x, y, z = centers[index]
            value = int(np.asarray(seg[x : x + 1, y : y + 1, z : z + 1].read().result()).ravel()[0])
            n_hit += value == root_id
            n_checked += 1

    share = n_hit / max(n_checked, 1)
    print(f"  {n_hit:,}/{n_checked:,} sampled centers hold their own root_id ({share:.1%})")
    if share < 0.90:
        print("  *** FAIL: the centers do not land on the cell. Wrong materialization, ")
        print("      wrong resolution, or a shifted window. Do not use these counts.")
    else:
        print("  OK -- the windows are centered on the right cell.")


def check_ranges(cells: list[dict]) -> None:
    print("\n=== 2. counts in range ===")
    counts = np.concatenate([c["voxel_count"] for c in cells])
    box = _box_voxels(int(cells[0]["box_size"]))
    n_zero = int((counts == 0).sum())
    print(f"  nodes            {len(counts):,}")
    print(f"  min / max        {counts.min():,} / {counts.max():,}   (box holds {box:,})")
    print(f"  zero-count nodes {n_zero:,} ({n_zero / len(counts):.2%})")
    if counts.min() < 0 or counts.max() > box:
        print("  *** FAIL: counts outside [0, box_size^3].")
    elif n_zero / len(counts) > 0.01:
        print("  *** SUSPECT: >1% of nodes see none of their own cell.")
    else:
        print("  OK")


def check_occupancy(cells: list[dict]) -> None:
    print("\n=== 3. occupancy is physically plausible ===")
    counts = np.concatenate([c["voxel_count"] for c in cells])
    clipped = np.concatenate([c["clipped"] for c in cells])
    box = _box_voxels(int(cells[0]["box_size"]))
    voxel_nm3 = int(cells[0]["voxel_volume_nm3"])

    interior = counts[~clipped]
    pct = 100.0 * interior / box
    print(f"  interior nodes   {len(interior):,}  ({clipped.sum():,} clipped at the volume edge)")
    for q in (5, 25, 50, 75, 95):
        value = np.percentile(interior, q)
        print(f"  p{q:<3d} {value:>12,.0f} voxels  {np.percentile(pct, q):>6.2f}%  "
              f"{value * voxel_nm3 / 1e9:>8.3f} um^3")
    median_pct = np.median(pct)
    if not 0.05 <= median_pct <= 25.0:
        print(f"  *** SUSPECT: median occupancy {median_pct:.3f}% is outside the range a ")
        print("      neurite through a 4 um box should produce.")
    else:
        print("  OK -- consistent with a neurite passing through the box.")


def check_against_thickness(cells: list[dict]) -> None:
    """Independent cross-check: mask volume vs. ray-cast dendrite radius."""
    print("\n=== 4. volume tracks independently measured dendrite radius ===")
    if not THICKNESS_DIR.exists():
        print(f"  skipped: no {THICKNESS_DIR}")
        return

    counts, radii = [], []
    for cell in cells:
        thickness_path = THICKNESS_DIR / f"{int(cell['root_id'])}.npz"
        if not thickness_path.exists():
            continue
        radius_nm = np.load(thickness_path)["radius_nm"]
        # The thickness cache is indexed by SKELETON vertex; orig_node_ids is the only
        # correct bridge from graph-node order back to it. Never a positional zip.
        node_radii = radius_nm[cell["orig_node_ids"]]
        measured = np.isfinite(node_radii) & ~cell["clipped"]
        counts.append(cell["voxel_count"][measured])
        radii.append(node_radii[measured])

    if not counts or sum(len(c) for c in counts) < 100:
        print("  skipped: fewer than 100 nodes with both a count and a measured radius")
        return

    counts = np.concatenate(counts).astype(np.float64)
    radii = np.concatenate(radii).astype(np.float64)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        rank_a = np.argsort(np.argsort(a)).astype(np.float64)
        rank_b = np.argsort(np.argsort(b)).astype(np.float64)
        rank_a -= rank_a.mean()
        rank_b -= rank_b.mean()
        return float(rank_a @ rank_b / np.sqrt((rank_a @ rank_a) * (rank_b @ rank_b)))

    rho = spearman(counts, radii)
    # Spearman only against radius, not radius^2: the correlation is rank-based and
    # squaring a positive quantity is monotonic, so the two are the same number by
    # construction. Reporting both would look like corroboration and be arithmetic.
    print(f"  nodes with both  {len(counts):,}")
    print(f"  spearman(count, radius)  {rho:+.3f}")
    print(f"  pearson(count, radius^2) {_pearson(counts, radii ** 2):+.3f}  "
          "(cross-section scales with r^2, so this is the linear form to expect)")
    if rho < 0.1:
        print("  *** SUSPECT: no relationship to an independent radius measurement.")
    else:
        print("  OK -- two unrelated pipelines agree on which nodes are thick.")


def main() -> int:
    cells = load_cache(CACHE_DIR)
    total_nodes = sum(len(c["voxel_count"]) for c in cells)
    print(f"cache      {CACHE_DIR}")
    print(f"cells      {len(cells):,}  ({total_nodes:,} nodes)")
    print(f"box        {int(cells[0]['box_size'])}^3 voxels at "
          f"{tuple(cells[0]['resolution_nm'])} nm, mat {int(cells[0]['mat_version'])}")

    check_center_voxel(cells)
    check_ranges(cells)
    check_occupancy(cells)
    check_against_thickness(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
