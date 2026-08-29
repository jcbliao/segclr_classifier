"""Validates the user's recollection of the mat_version=1718 label source
against live CAVE data, and checks overlap with the real segclr-db store's
root_ids (resnet_860b_reshuffled, 2193 cells).

segclr_db.cave.CELL_SUBSETS["cortical_neurons"] already encodes this exactly:
cell_type from `cell_type_multifeature_combo`, filtered to cells with
status_axon=True in `proofreading_status_and_strategy` (1718 isn't in either
override dict, so both fall back to their defaults). Reused here as a pure
function (query_cells), not through SegCLRWriter.

Run via sbatch (mit_normal -- CAVE query + a join against a 12M-row skeleton
table needs real memory, same lesson as scripts/explore_real_store.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import cave as cave_mod  # noqa: E402
from segclr_db.cave import CAVEConfig  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402

# minnie65_phase3_v1 needs CAVE "view" permission jcbliao's account doesn't have
# (403 on 2026-08-05). mat_version 1718 is also listed as available under the
# PUBLIC minnie65_public datastack (confirmed via explore_cave_alignment.py's
# get_versions() output), which this account already has working access to --
# try that first, since it may sidestep the permission gap entirely. Override
# with DATASTACK=minnie65_phase3_v1 once/if that permission is granted.
DATASTACK = os.environ.get("DATASTACK", "minnie65_public")
MAT_VERSION = 1718
STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"


def main() -> int:
    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set")
        return 2

    print(f"CELL_SUBSETS['cortical_neurons'] = {cave_mod.CELL_SUBSETS['cortical_neurons']}")

    config = CAVEConfig(datastack=DATASTACK, materialization_version=MAT_VERSION, token=token)
    client = config.build_client()

    print(f"\nquerying cortical_neurons subset at {DATASTACK} / mat_version {MAT_VERSION} ...")
    frame = cave_mod.query_cells(client, DATASTACK, MAT_VERSION, subsets=["cortical_neurons"])
    print(f"{len(frame)} rows, {frame['root_id'].nunique()} distinct root_ids")
    print(f"\ncell_type value_counts:\n{frame['label'].value_counts()}")
    print(f"\nsource_table: {frame['source_table'].unique().tolist()}")
    print(f"proofread_axon value_counts:\n{frame['proofread_axon'].value_counts(dropna=False)}")

    print("\n" + "=" * 70)
    print("overlap with the real embeddings store's root_ids")
    print("=" * 70)
    db = SegCLRDatabase(root=STORE_ROOT, dataset="microns")
    store_ids = set(int(r) for r in db.raw_sql("select distinct root_id from skeleton_manifest")["root_id"])
    label_ids = set(int(r) for r in frame["root_id"])
    overlap = store_ids & label_ids
    print(f"store root_ids: {len(store_ids)}")
    print(f"labeled (cortical_neurons) root_ids: {len(label_ids)}")
    print(f"overlap: {len(overlap)}")
    if overlap:
        overlap_frame = frame[frame["root_id"].isin(overlap)]
        print(f"\ncell_type value_counts within the overlap:\n{overlap_frame['label'].value_counts()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
