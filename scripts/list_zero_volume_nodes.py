"""Every node whose SegCLR window contains none of its own cell -> a CSV.

A zero count means: the 129^3 box around this skeleton vertex is fully inside the
segmentation, and not one of its 2.15M voxels belongs to the root_id the skeleton
says this vertex belongs to. That is the same skeleton/segmentation disagreement
as the one-voxel center misses (see CLAUDE.md, "Mask volume"), but total rather
than marginal, and it is rare enough to enumerate exactly: 69 of 12,130,814 nodes.

Worth looking at individually rather than filtering, because the candidate causes
are distinguishable by eye in Neuroglancer and have different implications:

  * a segmentation hole or a merge/split the skeleton predates,
  * a skeleton fragment whose root_id has drifted (proofreading mints new ids),
  * a genuinely tiny disconnected fragment sitting outside its own segment.

The CSV carries the coordinates in three forms so no conversion is needed at the
point of use: nm (the canonical form, straight from the graph cache's `pos`), the
32x32x40 voxel index the window was actually built from, and the 4x4x40 voxel
index MICrONS Neuroglancer links use. `clipped` is included because a clipped
window's zero has a mundane explanation (it was mostly padding) while an
unclipped one does not.

Run via sbatch (mit_quicktest is enough). Read-only.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "mask_volume_cache"
GRAPH_CACHE_DIR = REPO / "data" / "graph_cache"
MANIFEST_PATH = REPO / "data" / "manifest.json"
OUT_PATH = REPO / "results" / "zero_volume_nodes.csv"

#: MICrONS Neuroglancer's coordinate space, for pasteable positions.
NG_RESOLUTION_NM = np.array([4.0, 4.0, 40.0])


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())["cells"]
    rows = []
    n_nodes = n_cells = 0

    for path in sorted(CACHE_DIR.glob("*.npz")):
        with np.load(path) as npz:
            counts = npz["voxel_count"]
            n_nodes += len(counts)
            n_cells += 1
            zero = np.nonzero(counts == 0)[0]
            if not len(zero):
                continue
            root_id = int(npz["root_id"])
            orig_node_ids = npz["orig_node_ids"]
            centers = npz["center_vox"]
            clipped = npz["clipped"].astype(bool)
            resolution = npz["resolution_nm"]

        # pos is the canonical nm coordinate and the exact array the window
        # centers were derived from, so it is read from the graph cache rather
        # than reconstructed from center_vox (which truncated and cannot be
        # inverted without losing up to a voxel per axis).
        graph_path = GRAPH_CACHE_DIR / f"{root_id}.pt"
        if not graph_path.exists():
            print(f"WARNING: {root_id} has a volume cache but no graph cache; "
                  "coordinates omitted for its rows")
            pos = None
        else:
            pos = torch.load(graph_path, weights_only=False).pos.numpy()

        meta = manifest.get(str(root_id), {})
        for index in zero:
            xyz = pos[index] if pos is not None else np.full(3, np.nan)
            ng = xyz / NG_RESOLUTION_NM
            rows.append({
                "root_id": root_id,
                "cell_type": meta.get("cell_type", ""),
                "split": meta.get("split", ""),
                "graph_node_index": int(index),
                "skeleton_node_id": int(orig_node_ids[index]),
                "x_nm": float(xyz[0]),
                "y_nm": float(xyz[1]),
                "z_nm": float(xyz[2]),
                # The voxel the window was centered on, in the segmentation's own
                # 32x32x40 grid -- this is the index the count was taken at.
                "center_vox_x": int(centers[index][0]),
                "center_vox_y": int(centers[index][1]),
                "center_vox_z": int(centers[index][2]),
                # Pasteable into a MICrONS Neuroglancer position box.
                "ng_x": round(float(ng[0]), 1),
                "ng_y": round(float(ng[1]), 1),
                "ng_z": round(float(ng[2]), 1),
                "clipped": bool(clipped[index]),
                "seg_resolution_nm": "x".join(str(int(r)) for r in resolution),
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"no zero-volume nodes found across {n_cells:,} cells / {n_nodes:,} nodes")
        return 0

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_clipped = sum(r["clipped"] for r in rows)
    affected = sorted({r["root_id"] for r in rows})
    print(f"scanned  {n_cells:,} cells / {n_nodes:,} nodes")
    print(f"found    {len(rows)} zero-volume nodes across {len(affected)} cells")
    print(f"         {n_clipped} clipped (mundane: mostly padding), "
          f"{len(rows) - n_clipped} unclipped (the interesting ones)")
    print(f"wrote    {OUT_PATH}")

    print("\nper affected cell:")
    print(f"  {'root_id':>20s} {'cell_type':>12s} {'split':>6s} {'n_zero':>7s}")
    for rid in affected:
        sel = [r for r in rows if r["root_id"] == rid]
        print(f"  {rid:>20d} {sel[0]['cell_type']:>12s} {sel[0]['split']:>6s} "
              f"{len(sel):>7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
