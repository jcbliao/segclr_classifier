"""Which of the h5's 2442 cells already have a cached skeleton on disk, and
which need a fresh CAVE fetch? Run via sbatch (scripts/sbatch/check_h5_skeleton_coverage.sh)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

H5_PATH = "/orcd/compute/sdorkenw/001/collina/data/all_cells_aggregated_1718.h5"
REPO = Path(__file__).resolve().parent.parent
SKEL_CACHE = REPO / "data" / "skeleton_cache"

with h5py.File(H5_PATH, "r") as f:
    seg_ids = np.unique(f["seg_ids"][:])

h5_root_ids = set(int(s) for s in seg_ids)
cached_ids = set(int(p.stem) for p in SKEL_CACHE.glob("*.pkl"))

covered = h5_root_ids & cached_ids
missing = h5_root_ids - cached_ids

print(f"h5 cells: {len(h5_root_ids)}", flush=True)
print(f"cached skeletons on disk: {len(cached_ids)}", flush=True)
print(f"h5 cells already cached: {len(covered)}", flush=True)
print(f"h5 cells needing a fresh skeleton fetch: {len(missing)}", flush=True)

out_path = REPO / "data" / "h5_missing_skeleton_root_ids.json"
out_path.write_text(json.dumps(sorted(missing)))
print(f"wrote missing root_ids to {out_path}", flush=True)
print("done.", flush=True)
