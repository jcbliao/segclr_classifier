"""Follow-up to scripts/explore_db_cell_labels.py: the 217 cells segclr_db's
cell_labels table has that our old CAVE-query manifest didn't turned out to
ALL fail with "no covered nodes" when data/build_dataset_from_store.py tried
to build them (job 19772223) -- this checks a few directly to confirm why:
does resnet_860b_reshuffled have zero embedding rows for them (a real
data-availability gap, this experiment was likely only ever run on the
neuron-only subset), or something else.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402
from segclr_db.skeletons import SkeletonCache  # noqa: E402

STORE_ROOT = "/orcd/compute/sdorkenw/001/collina/segclr-db"
STORE_DATASET = "microns"
EXPERIMENT_ID = "resnet_860b_reshuffled"

SAMPLE_ROOT_IDS = [864691135941519361, 864691135579295749, 864691136120465944]


def main() -> int:
    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)
    skel_cache = SkeletonCache(store)

    for rid in SAMPLE_ROOT_IDS:
        try:
            r = db.get_embeddings(EXPERIMENT_ID, root_ids=rid)
            n_emb = len(r.node_ids)
        except Exception as e:  # noqa: BLE001
            n_emb = f"ERROR {type(e).__name__}: {e}"
        try:
            skel = skel_cache.get_skeleton(rid, fetch_if_missing=False)
            n_skel = len(skel)
        except Exception as e:  # noqa: BLE001
            n_skel = f"ERROR {type(e).__name__}: {e}"
        print(f"root_id={rid}  embedding_node_ids={n_emb}  skeleton_nodes={n_skel}")

    print("\n=== experiment metadata for resnet_860b_reshuffled ===")
    exp = db.experiment(EXPERIMENT_ID)
    print(exp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
