"""First look at the now-accessible, populated segclr-db store at
/orcd/compute/sdorkenw/001/segclr-db -- what experiment(s), how many
cells, are labels/splits already registered, does node_embeddings already
carry a clean (root_id, node_id) correspondence to skeleton_nodes. Read-only
(SegCLRDatabase, never SegCLRWriter). Run via sbatch (mit_quicktest).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db.database import SegCLRDatabase  # noqa: E402

ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
DATASET = "microns"


def main() -> int:
    db = SegCLRDatabase(root=ROOT, dataset=DATASET)

    print("=" * 70)
    print("1. tables in the store")
    print("=" * 70)
    print(db.tables().to_string(index=False))

    print()
    print("=" * 70)
    print("2. experiments")
    print("=" * 70)
    experiments = db.list_experiments()
    print(experiments.to_string(index=False))

    for exp_id in experiments["experiment_id"]:
        print(f"\n--- describe_experiment({exp_id!r}) ---")
        try:
            print(db.describe_experiment(exp_id).to_string())
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {e}")

        print(f"\n--- list_runs({exp_id!r}) ---")
        try:
            print(db.list_runs(exp_id).to_string(index=False))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {e}")

        print(f"\n--- list_agg_specs(experiment={exp_id!r}) ---")
        try:
            print(db.list_agg_specs(exp_id).to_string(index=False))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {e}")

    print()
    print("=" * 70)
    print("3. cells / labels / splits / hierarchies")
    print("=" * 70)
    try:
        cells = db.get_cells()
        print(f"get_cells(): {len(cells)} rows")
        print(cells.head(10).to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print(f"get_cells() FAILED: {type(e).__name__}: {e}")

    try:
        labels = db.get_labels()
        print(f"\nget_labels(): {len(labels)} rows")
        print(labels.head(10).to_string(index=False))
        if "label" in labels.columns:
            print(f"\nlabel value_counts:\n{labels['label'].value_counts()}")
    except Exception as e:  # noqa: BLE001
        print(f"get_labels() FAILED: {type(e).__name__}: {e}")

    try:
        print(f"\nlist_splits():\n{db.list_splits().to_string(index=False)}")
    except Exception as e:  # noqa: BLE001
        print(f"list_splits() FAILED: {type(e).__name__}: {e}")

    try:
        print(f"\nlist_hierarchies():\n{db.list_hierarchies().to_string(index=False)}")
    except Exception as e:  # noqa: BLE001
        print(f"list_hierarchies() FAILED: {type(e).__name__}: {e}")

    print()
    print("=" * 70)
    print("4. one real cell, end to end (if we have at least one experiment + cell)")
    print("=" * 70)
    try:
        exp_id = experiments["experiment_id"].iloc[0]
        # cells table is empty (no labels/splits registered in this store), so
        # pull a root_id straight from the skeleton manifest instead.
        skel_ids = db.raw_sql("select root_id from skeleton_manifest limit 5")
        root_id = int(skel_ids["root_id"].iloc[0])
        print(f"experiment={exp_id!r} root_id={root_id} (from skeleton_manifest, not cells)")

        result = db.get_embeddings(exp_id, root_ids=root_id, return_coords=True)
        print(f"raw node_embeddings: {result.embeddings.shape} dtype={result.embeddings.dtype}")
        print(f"  node_ids range: {result.node_ids.min()}..{result.node_ids.max()}")
        print(f"  coords shape: {None if result.coords is None else result.coords.shape}")

        completeness = db.check_completeness(exp_id, root_id)
        print(f"\ncompleteness: {completeness.to_dataframe().to_string(index=False)}")

        agg_specs = db.list_agg_specs(exp_id)
        if len(agg_specs):
            spec_id = agg_specs["agg_spec_id"].iloc[0]
            agg_result = db.get_embeddings(exp_id, root_ids=root_id, agg_spec_id=spec_id)
            print(f"\nagg_embeddings ({spec_id}): {agg_result.embeddings.shape}")

        skel = db.skeletons.get_skeleton(root_id, fetch_if_missing=False)
        print(f"\nskeleton: {len(skel)} nodes, {len(skel.edges)} edges")
        print(f"  node_embeddings node_id range vs skeleton node count: "
              f"emb max={result.node_ids.max()}  skeleton n_nodes={len(skel)}")
    except Exception as e:  # noqa: BLE001
        print(f"end-to-end check FAILED: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
