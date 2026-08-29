"""Does the store hold cells labeled with the glia classes the hierarchy
expects -- astrocyte, oligo, microglia, OPC -- and are any of them usable?

LAB_HIERARCHY_TREE's non_neuron branch is `glia: [astrocyte, oligo, microglia,
OPC]`, but data/manifest.json covers 18 granular classes, all under `neuron`.
So the four glia heads never receive gradient. This asks WHY, at the source,
and separates three different failure modes that look identical from the
manifest:

  (a) the label simply does not exist in cell_labels     -> nothing to ingest
  (b) labeled, but no embeddings under the experiment    -> re-run embed_cells
  (c) labeled and embedded, just not in our manifest     -> rebuild the dataset

Scans EVERY label_set in cell_labels, not just "cell_type", since a class
missing from one set may live in another. For each label it reports how many
cells have embeddings under EXPERIMENT_ID and how many have a cached skeleton
(both are prerequisites for data/build_dataset_from_store.py).

Read-only, store only -- no CAVE calls, so no CAVE_TOKEN needed.
Run via sbatch (mit_quicktest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

import pandas as pd  # noqa: E402

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402
from segclr_db.skeletons import SkeletonCache  # noqa: E402

from gnn.hierarchy import LAB_HIERARCHY_TREE  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
STORE_DATASET = "microns"
EXPERIMENT_ID = "resnet_860b_reshuffled"

GLIA_LABELS = LAB_HIERARCHY_TREE["non_neuron"]["non_neuron"]["glia"]

MAX_LISTED = 10


def main() -> int:
    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)

    # ---- every label_set, not just cell_type -----------------------------
    raw = st.scan(
        store, "cell_labels", columns=["root_id", "label_set", "label", "source_table"]
    ).to_pandas()
    print(f"cell_labels: {len(raw):,} rows, {raw['root_id'].nunique():,} distinct root_ids")
    print(f"label_sets present: {sorted(raw['label_set'].unique().tolist())}")
    print(f"source_tables: {sorted(raw['source_table'].dropna().unique().tolist())}")

    print("\n=== per (label_set, label) counts ===")
    counts = (
        raw.groupby(["label_set", "label"])["root_id"]
        .nunique()
        .reset_index(name="n_cells")
        .sort_values(["label_set", "n_cells"], ascending=[True, False])
    )
    with pd.option_context("display.max_rows", None):
        print(counts.to_string(index=False))

    # ---- the four glia classes specifically ------------------------------
    print(f"\n=== hierarchy's glia leaves: {GLIA_LABELS} ===")
    present = raw[raw["label"].isin(GLIA_LABELS)]
    missing = [lab for lab in GLIA_LABELS if lab not in set(raw["label"])]
    print(f"present in cell_labels: {sorted(set(present['label']))}")
    print(f"ABSENT from cell_labels entirely: {missing or 'none'}")
    if missing:
        # Near-misses matter: a class could be there under another spelling
        # (e.g. "oligodendrocyte" vs "oligo") rather than genuinely absent.
        others = sorted(set(raw["label"]) - set(GLIA_LABELS))
        print(f"  all other labels in the store, for spelling comparison: {others}")

    if present.empty:
        print("\nno glia-labeled cells at all -- nothing further to check.")
        return 0

    # ---- are the glia cells usable? --------------------------------------
    glia_ids = sorted({int(r) for r in present["root_id"]})
    embedded = {int(r) for r in db.cells_with_embeddings(EXPERIMENT_ID).tolist()}
    node_counts = SkeletonCache(store).node_counts(glia_ids)
    ours = {int(rid) for rid in json.loads(MANIFEST_PATH.read_text())["cells"]}

    label_of = {int(r.root_id): str(r.label) for r in present.itertuples()}
    rows = []
    for lab in sorted(set(present["label"])):
        ids = [r for r in glia_ids if label_of[r] == lab]
        rows.append(
            {
                "label": lab,
                "labeled": len(ids),
                "has_skeleton": sum(1 for r in ids if node_counts.get(r, 0) > 0),
                f"embedded_{EXPERIMENT_ID}": sum(1 for r in ids if r in embedded),
                "in_our_manifest": sum(1 for r in ids if r in ours),
            }
        )
    print(f"\n=== usability of glia-labeled cells (experiment={EXPERIMENT_ID}) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # Ready = labeled AND embedded AND has a skeleton, but not yet in the
    # manifest: exactly the cells a dataset rebuild would newly pick up.
    ready = [
        r for r in glia_ids
        if r in embedded and node_counts.get(r, 0) > 0 and r not in ours
    ]
    print(f"\n{len(ready)} glia cells are labeled + embedded + skeletonized but NOT in "
          f"data/manifest.json -- a dataset rebuild would add them.")
    for r in ready[:MAX_LISTED]:
        print(f"    {r}  {label_of[r]}  n_skel={node_counts.get(r, 0)}")
    if len(ready) > MAX_LISTED:
        print(f"    ... and {len(ready) - MAX_LISTED} more")

    blocked = [r for r in glia_ids if r not in embedded]
    print(f"\n{len(blocked)} glia cells are labeled but have no embeddings under "
          f"{EXPERIMENT_ID} -- these need an embed_cells run, not a rebuild.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
