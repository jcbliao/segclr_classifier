"""Are there new cells in the segclr-db store since data/manifest.json was built?

Read-only census, three questions in order:

  1. LABELS -- does `cell_labels` (label_set="cell_type") now name root_ids our
     manifest doesn't have? Also flags the reverse (manifest cells that dropped
     out of the label table) and any label that CHANGED on the overlap, since a
     relabel is as much a dataset change as a new cell.
  2. EMBEDDINGS -- of any newly labeled cells, how many actually have
     `resnet_860b_reshuffled` embedding rows? This is the gate that produced the
     current 2192-cell coverage (scripts/check_new_cells_embedding_coverage.py):
     cell_labels names non-neuron cells this experiment never embedded, so
     "newly labeled" and "newly usable" are different numbers. Uses
     db.cells_with_embeddings, which projects the root_id column only rather
     than materializing ~8 GB of vectors.
  3. SKELETONS -- of the newly usable cells, how many have a skeleton already
     ingested? Without one, build_dataset_from_store.py errors that cell out
     ("no skeleton in store") and the skeleton would have to be fetched from
     CAVE first (~10 req/min).

Also lists registered experiments, in case a NEW embedding experiment (not just
new cells under the old one) has appeared -- that would be a bigger change than
either of the above.

Run via sbatch (scripts/sbatch/check_new_cells.sh), CPU-only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402
from segclr_db.skeletons import SkeletonCache  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"
GRAPH_CACHE_DIR = REPO / "data" / "graph_cache"
WINDOW_DIR = REPO / "data" / "window_membership"

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
STORE_DATASET = "microns"
EXPERIMENT_ID = "resnet_860b_reshuffled"
LABEL_SET = "cell_type"
EXCLUDED_LABELS = {"ChC"}  # kept in sync with data/build_dataset_from_store.py

MAX_LISTED = 30


def _counts(labels: dict[int, str], ids) -> str:
    c = Counter(labels[r] for r in ids)
    return ", ".join(f"{lab}={n}" for lab, n in c.most_common())


def main() -> int:
    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)

    manifest = json.loads(MANIFEST_PATH.read_text())
    ours = {int(rid): info["cell_type"] for rid, info in manifest["cells"].items()}
    print(f"=== our dataset (data/manifest.json) ===")
    print(f"{len(ours)} cells, {len(set(ours.values()))} distinct cell_type values")
    print(f"manifest label_source={manifest.get('label_source')!r} "
          f"label_set={manifest.get('label_set')!r} "
          f"label_mat_versions={manifest.get('label_mat_versions')}")
    n_pt = len(list(GRAPH_CACHE_DIR.glob("*.pt"))) if GRAPH_CACHE_DIR.exists() else 0
    n_npz = len(list(WINDOW_DIR.glob("*.npz"))) if WINDOW_DIR.exists() else 0
    print(f"on disk: {n_pt} graph_cache/*.pt, {n_npz} window_membership/*.npz")

    # ---- 1. labels -------------------------------------------------------
    df = db.get_labels(label_set=LABEL_SET)
    mat_versions = sorted(int(v) for v in df["mat_version"].dropna().unique())
    store_labels_all = {int(r.root_id): str(r.label) for r in df.itertuples()}
    store_labels = {r: lab for r, lab in store_labels_all.items() if lab not in EXCLUDED_LABELS}
    n_excluded = len(store_labels_all) - len(store_labels)

    print(f"\n=== store cell_labels (label_set={LABEL_SET!r}) ===")
    print(f"{len(df)} rows, {len(store_labels_all)} distinct root_ids, "
          f"mat_versions={mat_versions}")
    print(f"{n_excluded} dropped by EXCLUDED_LABELS={sorted(EXCLUDED_LABELS)} "
          f"-> {len(store_labels)} eligible")
    print("per-label counts:")
    for lab, n in Counter(store_labels.values()).most_common():
        print(f"  {lab:<20} {n}")

    new_labeled = sorted(set(store_labels) - set(ours))
    gone = sorted(set(ours) - set(store_labels))
    relabeled = sorted(r for r in set(ours) & set(store_labels) if ours[r] != store_labels[r])

    print(f"\n=== label diff vs our manifest ===")
    print(f"NEW labeled cells (in store, not in manifest): {len(new_labeled)}")
    if new_labeled:
        print(f"  by label: {_counts(store_labels, new_labeled)}")
        for r in new_labeled[:MAX_LISTED]:
            print(f"    {r}  {store_labels[r]}")
        if len(new_labeled) > MAX_LISTED:
            print(f"    ... and {len(new_labeled) - MAX_LISTED} more")
    print(f"cells in our manifest but NOT in store cell_labels: {len(gone)}")
    for r in gone[:MAX_LISTED]:
        print(f"    {r}  was {ours[r]}")
    if len(gone) > MAX_LISTED:
        print(f"    ... and {len(gone) - MAX_LISTED} more")
    print(f"RELABELED on the overlap: {len(relabeled)}")
    for r in relabeled[:MAX_LISTED]:
        print(f"    {r}  {ours[r]} -> {store_labels[r]}")
    if len(relabeled) > MAX_LISTED:
        print(f"    ... and {len(relabeled) - MAX_LISTED} more")

    # ---- 2. embeddings ---------------------------------------------------
    print(f"\n=== embedding coverage, experiment {EXPERIMENT_ID!r} ===")
    embedded = set(int(r) for r in db.cells_with_embeddings(EXPERIMENT_ID).tolist())
    print(f"{len(embedded)} distinct root_ids have node_embeddings rows")
    print(f"  of our {len(ours)} manifest cells, {len(set(ours) & embedded)} are embedded")
    embedded_labeled_not_ours = sorted((set(store_labels) & embedded) - set(ours))
    print(f"  labeled + embedded but NOT in our manifest: {len(embedded_labeled_not_ours)}")
    if embedded_labeled_not_ours:
        print(f"    by label: {_counts(store_labels, embedded_labeled_not_ours)}")
        for r in embedded_labeled_not_ours[:MAX_LISTED]:
            print(f"      {r}  {store_labels[r]}")
        if len(embedded_labeled_not_ours) > MAX_LISTED:
            print(f"      ... and {len(embedded_labeled_not_ours) - MAX_LISTED} more")
    new_unembedded = sorted(set(new_labeled) - embedded)
    print(f"  newly labeled but NOT embedded (unusable as-is): {len(new_unembedded)}")
    if new_unembedded:
        print(f"    by label: {_counts(store_labels, new_unembedded)}")
    print(f"  embedded cells with no cell_type label at all: {len(embedded - set(store_labels_all))}")

    # ---- 3. skeletons ----------------------------------------------------
    print(f"\n=== skeleton availability for the newly usable cells ===")
    if embedded_labeled_not_ours:
        counts = SkeletonCache(store).node_counts(embedded_labeled_not_ours)
        have = [r for r in embedded_labeled_not_ours if counts.get(r, 0) > 0]
        print(f"{len(have)}/{len(embedded_labeled_not_ours)} have a skeleton already ingested")
        missing = [r for r in embedded_labeled_not_ours if counts.get(r, 0) == 0]
        for r in missing[:MAX_LISTED]:
            print(f"    no skeleton: {r}  {store_labels[r]}")
        if len(missing) > MAX_LISTED:
            print(f"    ... and {len(missing) - MAX_LISTED} more")
        if have:
            ns = sorted(counts[r] for r in have)
            print(f"    node counts: min={ns[0]} median={ns[len(ns) // 2]} max={ns[-1]} "
                  f"total={sum(ns)}")
    else:
        print("(none -- nothing new to check)")

    # ---- 4. has a new experiment appeared? --------------------------------
    print(f"\n=== registered experiments (is there a NEW embedding experiment?) ===")
    exps = db.list_experiments()
    cols = [c for c in ("experiment_id", "kind", "embedding_dim", "created_at", "description")
            if c in exps.columns]
    print(exps[cols].to_string(index=False, max_colwidth=60))

    print(f"\n=== other label_sets in cell_labels ===")
    print(st.distinct(store, "cell_labels", "label_set").to_pylist())

    # ---- verdict ----------------------------------------------------------
    print("\n=== verdict ===")
    if not new_labeled and not gone and not relabeled:
        print("No change: store cell_labels matches data/manifest.json exactly.")
    else:
        print(f"{len(new_labeled)} newly labeled, {len(embedded_labeled_not_ours)} of which "
              f"(plus any relabels) are embedded and thus buildable; "
              f"{len(gone)} manifest cells no longer labeled; {len(relabeled)} relabeled.")
        print("A rebuild means: data/build_dataset_from_store.py (resumable, skips cached "
              "cells) then data/build_window_membership.py for the new cells only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
