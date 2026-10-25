"""Summarize cached SegCLR crop mask sizes without reading segmentation volumes."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np


def describe(values):
    if not len(values):
        return {"n": 0}
    return dict(n=int(len(values)), mean=float(values.mean()),
                **{key: float(value) for key, value in zip(
                    ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"],
                    np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100]))})


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=repo / "data/mask_volume_cache")
    parser.add_argument("--output", type=Path, default=repo / "analysis/segmentation_mask_sizes")
    args = parser.parse_args()
    files = sorted(args.cache.glob("*.npz"))
    if not files:
        parser.error(f"no mask count caches in {args.cache}")
    counts, clipped, rows = [], [], []
    geometry = None
    for path in files:
        with np.load(path) as z:
            current = (int(z["box_size"]), tuple(z["resolution_nm"].tolist()),
                       int(z["mat_version"]), int(z["voxel_volume_nm3"]))
            if geometry is not None and current != geometry:
                raise ValueError(f"incompatible crop geometry/materialization: {path}")
            geometry = current
            c = z["voxel_count"].copy()
            flag = z["clipped"].astype(bool)
            if c.ndim != 1 or c.shape != flag.shape or np.any(c < 0) or np.any(c > current[0] ** 3):
                raise ValueError(f"invalid counts or flags: {path}")
            counts.append(c)
            clipped.append(flag)
            rows.append(dict(root_id=int(z["root_id"]), n_masks=len(c),
                             n_empty=int((c == 0).sum()), n_clipped=int(flag.sum()),
                             **{k: v for k, v in describe(c).items() if k != "n"}))
    c, flag = np.concatenate(counts), np.concatenate(clipped)
    box, resolution, version, voxel_volume = geometry
    summary = dict(cache=str(args.cache.resolve()), n_cells=len(files), box_size=box,
                   resolution_nm=resolution, mat_version=version,
                   crop_voxels=box ** 3, voxel_volume_um3=voxel_volume / 1e9,
                   n_empty=int((c == 0).sum()), n_clipped=int(flag.sum()),
                   all_masks_voxels=describe(c), unclipped_masks_voxels=describe(c[~flag]),
                   nonempty_unclipped_masks_voxels=describe(c[(~flag) & (c > 0)]),
                   occupancy_fraction=describe(c / box ** 3),
                   mask_volume_um3=describe(c * (voxel_volume / 1e9)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "per_cell.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    positive = c[c > 0]
    if len(positive):
        low, high = positive.min(), positive.max()
        bins = np.geomspace(low, max(high, low + 1), 81)
        axes[0].hist(positive, bins=bins)
        axes[0].set_xscale("log")
    axes[0].set(xlabel="Masked voxels per crop (nonempty masks)", ylabel="Number of crops")
    axes[1].hist(100 * c / box ** 3, bins=100)
    axes[1].set(xlabel="Crop occupied by mask (%)", ylabel="Number of crops")
    fig.suptitle(f"{len(c):,} masks across {len(files):,} cells; {int((c == 0).sum()):,} empty")
    fig.savefig(args.output / "mask_size_distribution.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
