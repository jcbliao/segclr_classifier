"""Does cell_type_multifeature_combo have more classes than the 19 we see
after filtering to the cortical_neurons subset (status_axon=True in
proofreading_status_and_strategy)? Queries the raw table directly, the same
way segclr_db.cave._query_cortical does internally before its proofreading
join/filter, and diffs against our current label set.

Run via sbatch (mit_normal, real memory for consistency with other CAVE
queries against this table).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db.cave import CAVEConfig  # noqa: E402

from data.dataset import load_manifest  # noqa: E402

DATASTACK = "minnie65_public"
MAT_VERSION = 1718
CELL_TYPE_TABLE = "cell_type_multifeature_combo"
PROOFREADING_TABLE = "proofreading_status_and_strategy"


def main() -> int:
    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set")
        return 2
    config = CAVEConfig(datastack=DATASTACK, materialization_version=MAT_VERSION, token=token)
    client = config.build_client()

    print(f"querying full {CELL_TYPE_TABLE} at mat_version {MAT_VERSION} (no proofreading filter) ...")
    types = client.materialize.query_table(
        CELL_TYPE_TABLE, materialization_version=MAT_VERSION, split_positions=True,
        desired_resolution=[1, 1, 1],
    )
    print(f"{len(types)} total rows, {types['pt_root_id'].nunique()} distinct root_ids")
    print(f"\nfull cell_type value_counts ({types['cell_type'].nunique()} distinct classes):")
    print(types["cell_type"].value_counts(dropna=False))

    manifest = load_manifest()
    our_classes = {info["cell_type"] for info in manifest["cells"].values()}
    print(f"\nour current dataset: {len(our_classes)} classes: {sorted(our_classes)}")

    all_classes = set(types["cell_type"].dropna().unique())
    missing = all_classes - our_classes
    print(f"\nclasses in {CELL_TYPE_TABLE} but NOT in our dataset ({len(missing)}):")
    for c in sorted(missing):
        n = (types["cell_type"] == c).sum()
        print(f"  {c}: {n} cells in the raw table")

    print(f"\nclasses in our dataset but not in the raw table pull: {sorted(our_classes - all_classes)}")

    print("\n" + "=" * 70)
    print("why the missing ones dropped out: proofread_axon coverage")
    print("=" * 70)
    proof = client.materialize.query_table(PROOFREADING_TABLE, materialization_version=MAT_VERSION)
    proof_axon_true = set(proof.loc[proof["status_axon"].astype(bool), "pt_root_id"])
    for c in sorted(missing):
        ids = set(types.loc[types["cell_type"] == c, "pt_root_id"])
        n_proofread = len(ids & proof_axon_true)
        print(f"  {c}: {len(ids)} total, {n_proofread} with proofread axon (status_axon=True)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
