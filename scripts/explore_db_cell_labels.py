"""Re-check segclr_db's registered cell_labels/label_hierarchies/splits now
that the store's permissions were opened up further (2026-08-06) --
data/dims/cell_labels.lance and data/dims/cells.lance now show a Aug 5 17:04
mtime, after the whole-store chmod o+r noted in CLAUDE.md. Earlier in this
project cell_labels was found either empty or permission-blocked; this
re-checks with a live read rather than assuming either finding still holds.

Prints: (1) available hierarchy_ids + label_sets in cell_labels, (2) total
labeled root_ids + per-label counts, (3) any already-registered split_ids,
(4) overlap + label agreement against our current CAVE-sourced
data/manifest.json labels (built by data/build_dataset_from_store.py).

Run via sbatch, read-only (SegCLRDatabase only, no SegCLRWriter).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402

STORE_ROOT = "/orcd/compute/sdorkenw/001/collina/segclr-db"
STORE_DATASET = "microns"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.json"


def main() -> int:
    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)

    print("=== cell_labels: label_set values ===")
    label_sets = st.distinct(store, "cell_labels", "label_set").to_pylist()
    print(label_sets)

    print("\n=== cell_labels: raw row count + distinct root_ids ===")
    raw = st.scan(store, "cell_labels").to_pandas()
    print(f"{len(raw)} rows, {raw['root_id'].nunique()} distinct root_ids")
    print(raw.head(10))

    print("\n=== cell_labels: per-label counts (all label_sets pooled) ===")
    print(raw["label"].value_counts())

    print("\n=== label_hierarchies: available hierarchy_ids ===")
    try:
        hier_ids = st.distinct(store, "label_hierarchies", "hierarchy_id").to_pylist()
        print(hier_ids)
    except Exception as e:  # noqa: BLE001
        print(f"  (failed: {e})")

    print("\n=== splits: any already-registered split_ids ===")
    try:
        split_ids = st.distinct(store, "splits", "split_id").to_pylist()
        print(split_ids)
        if split_ids:
            sm = st.scan(store, "split_members").to_pandas()
            print(sm.groupby(["split_id", "which"]).size())
    except Exception as e:  # noqa: BLE001
        print(f"  (failed: {e})")

    print("\n=== overlap with our current CAVE-sourced manifest.json ===")
    manifest = json.loads(MANIFEST_PATH.read_text())
    our_labels = {int(rid): info["cell_type"] for rid, info in manifest["cells"].items()}
    db_labels = {}
    for row in raw.itertuples():
        db_labels.setdefault(int(row.root_id), []).append((row.label_set, row.label))

    overlap = set(our_labels) & set(db_labels)
    print(f"our manifest: {len(our_labels)} cells; db cell_labels: {len(db_labels)} distinct root_ids; overlap: {len(overlap)}")

    n_checked, n_match = 0, 0
    mismatches = []
    for rid in list(overlap)[:2193]:
        n_checked += 1
        db_vals = {lab for _, lab in db_labels[rid]}
        if our_labels[rid] in db_vals:
            n_match += 1
        else:
            mismatches.append((rid, our_labels[rid], db_vals))
    print(f"label agreement on overlap: {n_match}/{n_checked}")
    if mismatches:
        print(f"first 10 mismatches: {mismatches[:10]}")

    print("\n=== cells only in db, not our manifest (first 20) ===")
    only_db = set(db_labels) - set(our_labels)
    print(f"{len(only_db)} cells")
    for rid in list(only_db)[:20]:
        print(f"  {rid}: {db_labels[rid]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
