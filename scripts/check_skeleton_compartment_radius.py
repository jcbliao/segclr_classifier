"""One-off check: do our already-cached Skeleton objects (data/skeleton_cache/*.pkl,
exported by data/build_dataset_from_store.py from the segclr_db store) actually
carry populated compartment/radius fields, or just the zero/NaN defaults
segclr_db.results.Skeleton falls back to when CAVE's response omitted them?
Determines whether dendrite-thickness work needs a fresh live CAVE skeleton
fetch for compartment labels, or can reuse what's already cached.

Run via sbatch (mit_quicktest -- read-only, no GPU, no CAVE call):
    sbatch scripts/sbatch/check_skeleton_compartment_radius.sh
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

SKELETON_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "skeleton_cache"


def main() -> int:
    pt_files = sorted(SKELETON_CACHE_DIR.glob("*.pkl"))[:5]
    if not pt_files:
        print(f"no .pkl files found under {SKELETON_CACHE_DIR}")
        return 1

    for path in pt_files:
        with open(path, "rb") as f:
            skel = pickle.load(f)
        n = len(skel.coords)
        comp = skel.compartments
        rad = skel.radii
        comp_nonzero = int((comp != 0).sum()) if comp is not None else -1
        rad_valid = int(np.isfinite(rad).sum()) if rad is not None else -1
        rad_range = f"({np.nanmin(rad):.1f}, {np.nanmax(rad):.1f})nm" if rad_valid else "n/a"
        print(
            f"{path.name}: n_vertices={n}  "
            f"compartments: nonzero={comp_nonzero}/{n} unique={sorted(set(comp.tolist())) if comp is not None else None}  "
            f"radii: finite={rad_valid}/{n} range={rad_range}"
        )
        print(f"  skeleton_version={getattr(skel, 'skeleton_version', None)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
