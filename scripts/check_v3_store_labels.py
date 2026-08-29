"""What is in the NEW v3 store at /orcd/compute/sdorkenw/001/segclr-db, and
does it finally carry the glia classes the hierarchy expects?

Context: upstream segclr_db moved to SCHEMA_VERSION 3 and a new store appeared
alongside the v2 one this project has been reading. Upstream's commit log says
"Updated cell type table to include custom non neurons", which is exactly the
gap scripts/check_glia_label_coverage.py measured against the v2 store (oligo
and OPC absent entirely; astrocyte/microglia labeled but absent from our
manifest).

This deliberately does NOT go through SegCLRDatabase/open_store. Our vendored
clone is pinned at v2 and `store.open_store` raises SchemaVersionError on a v3
store by design, so reading through the library would require pulling the clone
forward -- which would simultaneously make the v2 store unreadable and break
every other check_*.py. For a read-only question about one table, opening the
Lance dataset directly is the smaller move; it is a diagnostic, not a new data
path. Nothing here writes.

Run via sbatch (mit_quicktest). No CAVE token needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lance  # noqa: E402
import pandas as pd  # noqa: E402

from gnn.hierarchy import LAB_HIERARCHY_TREE  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"

V3_ROOT = Path("/orcd/compute/sdorkenw/001/segclr-db/microns")
V2_ROOT = Path("/orcd/compute/sdorkenw/001/collina/segclr-db/microns")

GLIA_LABELS = LAB_HIERARCHY_TREE["non_neuron"]["non_neuron"]["glia"]
MAX_LISTED = 10


def load(root: Path, rel: str, columns: list[str] | None = None) -> pd.DataFrame:
    return lance.dataset(str(root / rel)).to_table(columns=columns).to_pandas()


def main() -> int:
    for name, root in (("v3 (new)", V3_ROOT), ("v2 (current)", V2_ROOT)):
        meta = json.loads((root / "meta.json").read_text())
        print(f"{name}: {root}")
        print(f"    schema_version={meta['schema_version']} created={meta.get('created_ts')} "
              f"datastack={meta.get('datastack')} mat_version={meta.get('mat_version')}")

    labels = load(V3_ROOT, "dims/cell_labels.lance",
                  ["root_id", "label_set", "label", "source_table", "mat_version"])
    print(f"\n=== v3 cell_labels: {len(labels):,} rows, "
          f"{labels['root_id'].nunique():,} distinct root_ids ===")
    print(f"label_sets: {sorted(labels['label_set'].unique().tolist())}")
    print(f"mat_versions: {sorted(labels['mat_version'].dropna().unique().tolist())}")
    print(f"source_tables: {sorted(labels['source_table'].dropna().unique().tolist())}")

    print("\n=== per (label_set, label) counts ===")
    counts = (
        labels.groupby(["label_set", "label"])["root_id"]
        .nunique().reset_index(name="n_cells")
        .sort_values(["label_set", "n_cells"], ascending=[True, False])
    )
    with pd.option_context("display.max_rows", None):
        print(counts.to_string(index=False))

    # ---- the hierarchy's glia leaves, and what changed vs the v2 store ----
    v2_labels = load(V2_ROOT, "dims/cell_labels.lance", ["root_id", "label"])
    v2_set, v3_set = set(v2_labels["label"]), set(labels["label"])
    print(f"\n=== labels gained vs the v2 store: {sorted(v3_set - v2_set) or 'none'} ===")
    print(f"=== labels lost vs the v2 store:   {sorted(v2_set - v3_set) or 'none'} ===")

    print(f"\n=== hierarchy's glia leaves: {GLIA_LABELS} ===")
    print(f"present in v3: {sorted(l for l in GLIA_LABELS if l in v3_set) or 'none'}")
    print(f"ABSENT from v3: {sorted(l for l in GLIA_LABELS if l not in v3_set) or 'none'}")

    # ---- are the glia cells usable here? ---------------------------------
    # Embedded = has at least one node_embeddings row in this store; skeleton =
    # has a skeleton_manifest row. Both are prerequisites for
    # data/build_dataset_from_store.py, and either can be the blocker.
    emb_ids = set(
        load(V3_ROOT, "embeddings/node_embeddings/d64.lance", ["root_id"])["root_id"]
        .unique().tolist()
    )
    skel_ids = set(
        load(V3_ROOT, "skeletons/skeleton_manifest.lance", ["root_id"])["root_id"]
        .unique().tolist()
    )
    ours = {int(rid) for rid in json.loads(MANIFEST_PATH.read_text())["cells"]}
    print(f"\nv3 store: {len(emb_ids):,} cells with d64 embeddings, "
          f"{len(skel_ids):,} with cached skeletons")

    rows = []
    for lab in sorted(v3_set):
        ids = {int(r) for r in labels.loc[labels["label"] == lab, "root_id"]}
        rows.append({
            "label": lab,
            "labeled": len(ids),
            "has_skeleton": len(ids & skel_ids),
            "embedded": len(ids & emb_ids),
            "ready_and_new": len((ids & emb_ids & skel_ids) - ours),
        })
    summary = pd.DataFrame(rows).sort_values("labeled", ascending=False)
    print("\n=== per-label usability in the v3 store ===")
    print(summary.to_string(index=False))
    print(f"\ntotal labeled+embedded+skeletonized but NOT in our manifest: "
          f"{int(summary['ready_and_new'].sum()):,}")

    glia_rows = summary[summary["label"].isin(GLIA_LABELS)]
    if not glia_rows.empty:
        print(f"\nof which glia: {int(glia_rows['ready_and_new'].sum()):,}")

    # ---- what experiments does this store define? ------------------------
    exps = load(V3_ROOT, "registry/experiments.lance")
    cols = [c for c in ("experiment_id", "kind", "embedding_dim", "created_ts") if c in exps]
    print(f"\n=== registry/experiments ({len(exps)} rows) ===")
    print(exps[cols].to_string(index=False) if cols else exps.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
