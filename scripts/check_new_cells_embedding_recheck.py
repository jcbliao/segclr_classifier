"""Disambiguates one finding from scripts/check_new_cells.py (job 20109431):
124 of the 217 labeled-but-unbuilt glia/thalamocortical cells DO have
resnet_860b_reshuffled embedding rows now, contradicting CLAUDE.md's
"zero embedding rows for non-neurons" note.

That note came from scripts/check_new_cells_embedding_coverage.py, which
sampled only 3 root_ids. Two readings: (a) the store gained embeddings for
these cells since, or (b) those 3 happened to fall in the 93 that still have
none. This re-checks exactly those 3 ids, plus work_units timestamps for the
non-neuron cells, which date the ingest either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
STORE_DATASET = "microns"
EXPERIMENT_ID = "resnet_860b_reshuffled"

SAMPLE_ROOT_IDS = [864691135941519361, 864691135579295749, 864691136120465944]


def main() -> int:
    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)

    print("=== the 3 cells the earlier check sampled ===")
    for rid in SAMPLE_ROOT_IDS:
        try:
            r = db.get_embeddings(EXPERIMENT_ID, root_ids=rid)
            print(f"  {rid}: {len(r.node_ids)} embedding rows")
        except Exception as e:  # noqa: BLE001
            print(f"  {rid}: ERROR {type(e).__name__}: {e}")

    print("\n=== work_units for node_embeddings: status counts ===")
    wu = st.scan(store, "work_units", filter="table = 'node_embeddings'").to_pandas()
    print(f"{len(wu)} rows; columns: {list(wu.columns)}")
    if "status" in wu.columns:
        print(wu["status"].value_counts())

    ts_col = next((c for c in ("finished_ts", "started_ts") if c in wu.columns), None)
    if ts_col:
        print(f"\n=== ingest timeline by {ts_col} (per month) ===")
        print(wu[ts_col].dt.to_period("M").value_counts().sort_index())

        print("\n=== timeline for the 124 newly-embedded (glia/TC) cells ===")
        labels = db.get_labels(label_set="cell_type")
        non_neuron = {int(r.root_id) for r in labels.itertuples()
                      if str(r.label) in {"astrocyte", "microglia", "thalamocortical"}}
        sub = wu[wu["root_id"].astype("int64").isin(non_neuron)]
        print(f"{len(sub)} work_unit rows for {len(non_neuron)} non-neuron labeled cells")
        if len(sub):
            print(sub[ts_col].dt.to_period("M").value_counts().sort_index())
            if "status" in sub.columns:
                print(sub["status"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
