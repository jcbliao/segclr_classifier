"""Smoke test #2: confirm CAVE access works for the public MICrONS materialization
that the ground-truth label table (`labeled_cell_m343_df_221011b.feather`) is
keyed against, and that a fetched CAVE skeleton's node coordinates line up
spatially with the public SegCLR embedding release's
`microns_nm_coord_public_offset_v343` xyz coordinates for the same root_id.

This is the crux of the ingestion design in data/ingest_public_microns.py: we
assign each embedding row to the *nearest* CAVE skeleton node by xyz distance
(Google's internal skeletonization node_id is not the same index space as
CAVE's), so it matters that the residual nearest-neighbor distance is small
(sub-micron) and not, say, off by a coordinate-frame bug (which would show up
as distances of many microns).

Run via sbatch (mit_quicktest) -- see scripts/sbatch/explore_cave_alignment.sh.
Uses a throwaway skeleton-cache store under data/_smoketest_store, not the
real ingestion store.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))
from segclr_db import store as st  # noqa: E402
from segclr_db.cave import CAVEConfig  # noqa: E402
from segclr_db.skeletons import SkeletonCache  # noqa: E402

from data import public_reader as pr  # noqa: E402

# Candidate (datastack, mat_version) pairs to try, most-likely first. The
# bucket path `microns_nm_coord_public_offset_v343` and the label filename
# `..._m343_df...` both point at materialization 343 of the public MICrONS
# datastack; "minnie65_public" is CAVE's conventional name for that public
# release datastack (as opposed to "minnie65_phase3_v1", the internal one
# referenced elsewhere in this repo's examples).
CANDIDATES = [
    ("minnie65_public", 343),
    ("minnie65_public_v343", 343),
    ("minnie65_phase3_v1", 343),
]

TEST_ROOT_ID = 864691135491639135  # first row of the real label table

STORE_ROOT = Path(__file__).resolve().parent.parent / "data" / "_smoketest_store"


def main() -> int:
    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set in environment -- see sbatch/explore_cave_alignment.sh")
        return 2

    working_config = None
    client = None
    for datastack, mat_version in CANDIDATES:
        print(f"trying datastack={datastack!r} mat_version={mat_version} ...")
        config = CAVEConfig(datastack=datastack, materialization_version=mat_version, token=token)
        try:
            client = config.build_client()
            # Cheap calls to confirm the datastack/version actually resolve.
            versions = client.materialize.get_versions()
            print(f"  OK -- available materialization versions: {sorted(versions)}")
            if mat_version not in versions:
                print(f"  WARNING: {mat_version} not in available versions, trying anyway")
            working_config = config
            break
        except Exception as e:  # noqa: BLE001 -- exploratory script
            print(f"  FAILED -- {type(e).__name__}: {e}")

    if working_config is None:
        print("\nno candidate datastack worked; need the correct datastack name from the user")
        return 1

    print(f"\nusing datastack={working_config.datastack} mat_version={working_config.materialization_version}")

    print("\nfetching one real CAVE skeleton ...")
    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    store = st.init_store(STORE_ROOT, dataset="microns") if not (STORE_ROOT / "microns").exists() \
        else st.open_store(STORE_ROOT, "microns")
    cache = SkeletonCache(store, cave_config=working_config)
    report = cache.ensure_cached([TEST_ROOT_ID])
    print(f"  {report.summary()}")
    skeleton = cache.get_skeleton(TEST_ROOT_ID, fetch_if_missing=False)
    print(f"  skeleton: {len(skeleton)} nodes")
    print(f"  coord range: {skeleton.coords.min(axis=0)} .. {skeleton.coords.max(axis=0)}")

    print("\nfetching public embeddings for the same cell ...")
    fs = pr.get_public_filesystem()
    cell = pr.get_raw_cell_embeddings(
        TEST_ROOT_ID, fs, data_key="microns_nm_coord_public_offset_v343"
    )
    print(f"  {cell.embeddings.shape[0]} embedding rows, dim={cell.embeddings.shape[1]}")
    print(f"  xyz range: {cell.xyz_nm.min(axis=0)} .. {cell.xyz_nm.max(axis=0)}")

    print("\nnearest-neighbor alignment check (embedding xyz -> nearest skeleton node) ...")
    tree = cKDTree(skeleton.coords)
    dist, _ = tree.query(cell.xyz_nm)
    print(f"  n={len(dist)}  median={np.median(dist):.1f}nm  mean={dist.mean():.1f}nm  "
          f"p95={np.percentile(dist, 95):.1f}nm  max={dist.max():.1f}nm")
    print(
        "  (expect median well under 1000nm if coordinate frames genuinely match; "
        "a median of many microns means the offset/scale is wrong)"
    )

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
