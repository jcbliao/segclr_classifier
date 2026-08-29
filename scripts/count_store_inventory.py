"""A plain census of the store: how many cells, how many embedding rows, at
each stage from "has a skeleton" down to "is in our training set".

Written because the numbers in this thread came from several different sources
(a job log, work_units, cells_with_embeddings) and needed to be put on one
footing. Also resolves an off-by-one: cells_with_embeddings reported 84,304
distinct root_ids while work_units held 84,303 rows -- either one cell was
written without a work record, or one cell has two work records.

Run via sbatch (mit_normal). Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
STORE_DATASET = "microns"
EXPERIMENT_ID = "resnet_860b_reshuffled"
LABEL_SET = "cell_type"
EXCLUDED_LABELS = {"ChC"}


def main() -> int:
    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)

    # ---- skeletons -------------------------------------------------------
    skel = st.scan(store, "skeleton_manifest", columns=["root_id", "n_nodes"]).to_pandas()
    print("=== skeletons ===")
    print(f"cells with a cached skeleton : {len(skel):,}")
    print(f"skeleton nodes across them   : {int(skel['n_nodes'].sum()):,}")

    # ---- embeddings ------------------------------------------------------
    n_emb_rows = st.count_rows(store, "node_embeddings", dim=64)
    embedded = db.cells_with_embeddings(EXPERIMENT_ID)
    wu = st.scan(store, "work_units", filter="table = 'node_embeddings'",
                 columns=["root_id", "status", "n_rows"]).to_pandas()

    print("\n=== embeddings (experiment resnet_860b_reshuffled, d64) ===")
    print(f"embedding ROWS (one per node): {n_emb_rows:,}")
    print(f"cells with >=1 embedding row : {len(embedded):,}")
    print(f"work_unit rows               : {len(wu):,}")
    print(f"work_unit distinct root_ids  : {wu['root_id'].nunique():,}")
    print(f"sum of work_unit n_rows      : {int(wu['n_rows'].sum()):,}")
    dupes = wu[wu.duplicated("root_id", keep=False)]
    print(f"cells with >1 work record    : {dupes['root_id'].nunique()}")
    if len(dupes):
        print(dupes.sort_values("root_id").head(6).to_string(index=False))
    only_emb = set(int(x) for x in embedded) - set(int(x) for x in wu["root_id"])
    only_wu = set(int(x) for x in wu["root_id"]) - set(int(x) for x in embedded)
    print(f"embedded but no work record  : {len(only_emb)} {sorted(only_emb)[:5]}")
    print(f"work record but no embeddings: {len(only_wu)} {sorted(only_wu)[:5]}")

    mean_nodes = n_emb_rows / max(len(embedded), 1)
    print(f"mean embedded nodes per cell : {mean_nodes:,.0f}")

    # ---- labels and our dataset ------------------------------------------
    df = db.get_labels(label_set=LABEL_SET)
    all_labeled = {int(r.root_id) for r in df.itertuples()}
    labeled = {int(r.root_id) for r in df.itertuples() if str(r.label) not in EXCLUDED_LABELS}
    ours = {int(rid) for rid in json.loads(MANIFEST_PATH.read_text())["cells"]}
    emb_set = {int(x) for x in embedded}

    print("\n=== labels ===")
    print(f"cells with a cell_type label : {len(all_labeled):,} "
          f"({len(labeled):,} after dropping {sorted(EXCLUDED_LABELS)})")
    print(f"labeled AND embedded         : {len(labeled & emb_set):,}")
    print(f"labeled, NOT embedded        : {len(labeled - emb_set):,}")
    print(f"embedded, NOT labeled        : {len(emb_set - all_labeled):,}")

    print("\n=== our training set ===")
    print(f"cells in data/manifest.json  : {len(ours):,}")
    print(f"  all embedded?              : {len(ours & emb_set) == len(ours)}")
    n_our_rows = int(wu[wu['root_id'].astype('int64').isin(ours)]['n_rows'].sum())
    print(f"embedding rows for those     : {n_our_rows:,}")
    print(f"buildable but not yet built  : {len(labeled & emb_set - ours):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
