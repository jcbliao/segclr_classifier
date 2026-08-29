"""Were the 93 labeled-but-unembedded cells inside the embed_cells sweep's
selection, or excluded from it?

Context: scripts/check_new_cells.py found 217 labeled cells missing from our
manifest -- 124 now embedded under resnet_860b_reshuffled, 93 with zero
embedding rows. Those embeddings came from the user's own embed_cells job
19886373 (~/projects/segclr/scripts/submit_embed_cells.sh), which selected
cells as:

    root_ids  = cache.cached_root_ids()                          # 192,984
    is_latest = client.chunkedgraph.is_latest_roots(root_ids,
                    client.materialize.get_timestamp())
    selected  = root_ids[is_latest]                              # 184,532

and got through ~84k of those 184,532 before hitting the 12h wall. So a cell
with no embeddings is in exactly one of three states, and they mean different
things:

  (a) no cached skeleton      -> never a candidate; needs skeleton ingest first
  (b) cached but not latest   -> deliberately filtered out as a stale root_id;
                                 re-running the sweep will never pick it up
  (c) cached, latest, no work_unit row -> simply not reached before the wall;
                                 resubmitting the sweep covers it

This reports which, per cell. Read-only apart from live CAVE lookups.
Run via sbatch (mit_quicktest); needs CAVE_TOKEN exported by the wrapper.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402
from segclr_db.skeletons import SkeletonCache  # noqa: E402

from segclr_db.cave import CAVEConfig  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
STORE_DATASET = "microns"
EXPERIMENT_ID = "resnet_860b_reshuffled"
LABEL_SET = "cell_type"
EXCLUDED_LABELS = {"ChC"}

# The sweep's own CAVE settings (~/projects/segclr/machine_config.yaml), NOT
# this repo's minnie65_public/mat-343 default -- see the is_latest_roots block.
CAVE_DATASTACK = "minnie65_phase3_v1"
CAVE_MAT_VERSION = 1718

MAX_LISTED = 20


def main() -> int:
    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set")
        return 2

    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)

    ours = {int(rid) for rid in json.loads(MANIFEST_PATH.read_text())["cells"]}
    df = db.get_labels(label_set=LABEL_SET)
    labels = {int(r.root_id): str(r.label) for r in df.itertuples()
              if str(r.label) not in EXCLUDED_LABELS}
    embedded = {int(r) for r in db.cells_with_embeddings(EXPERIMENT_ID).tolist()}

    unembedded = sorted(set(labels) - ours - embedded)
    print(f"{len(unembedded)} labeled cells not in our manifest and with no embeddings")
    print(f"  by label: {dict(Counter(labels[r] for r in unembedded).most_common())}")

    # ---- (a) does the store have a cached skeleton? ----------------------
    counts = SkeletonCache(store).node_counts(unembedded)
    has_skel = {r for r in unembedded if counts.get(r, 0) > 0}
    print(f"\ncached skeleton in the store: {len(has_skel)}/{len(unembedded)}")

    # For scale: how many skeletons does the cache hold overall? embed_cells
    # logged 192,984 -- if that still matches, the cache has not moved since.
    all_cached = st.scan(store, "skeleton_manifest", columns=["root_id"]).num_rows
    print(f"  (skeleton_manifest holds {all_cached:,} cells; embed_cells logged 192,984)")

    # ---- (c) was an attempt ever recorded? -------------------------------
    wu = st.scan(
        store,
        "work_units",
        filter="table = 'node_embeddings'",
        columns=["root_id", "status"],
        root_ids=unembedded,
    ).to_pandas()
    print(f"\nwork_unit rows for these cells: {len(wu)}")
    if len(wu):
        print(wu["status"].value_counts())
    attempted = {int(r) for r in wu["root_id"]} if len(wu) else set()

    # ---- (b) still a latest root? ---------------------------------------
    # Must match the sweep's OWN client, not this repo's default: embed_cells
    # built it from ~/projects/segclr/machine_config.yaml, i.e. datastack
    # minnie65_phase3_v1 at materialization 1718. This repo's
    # cave_skeletons.default_cave_config is minnie65_public at mat 343, whose
    # get_timestamp() is 2022-02-24 -- against which essentially every current
    # root_id reads as "not latest", which would fake a decisive answer.
    client = CAVEConfig(
        datastack=CAVE_DATASTACK, materialization_version=CAVE_MAT_VERSION, token=token
    ).build_client()
    ts = client.materialize.get_timestamp()
    print(f"\nCAVE client: datastack={CAVE_DATASTACK} mat_version={CAVE_MAT_VERSION}")
    print(f"is_latest_roots at materialize.get_timestamp() = {ts}")

    def _latest(ids: list[int]) -> set[int]:
        if not ids:
            return set()
        res = client.chunkedgraph.is_latest_roots(ids, ts)
        vals = list(res) if not isinstance(res, dict) else [res[r] for r in ids]
        return {r for r, ok in zip(ids, vals, strict=True) if ok}

    # Positive control: the 124 cells that DID get embedded necessarily passed
    # the sweep's is_latest filter. If they come back stale too, the filter is
    # not what separates the two groups and the verdict below is meaningless.
    control = sorted((set(labels) & embedded) - ours)
    control_latest = _latest(control)
    print(f"  CONTROL (the {len(control)} embedded cells): "
          f"{len(control_latest)}/{len(control)} latest")

    latest = _latest(unembedded)
    print(f"  the {len(unembedded)} unembedded cells: {len(latest)}/{len(unembedded)} latest")
    if control and len(control_latest) < len(control):
        print("  WARNING: control cells read as stale -- filter is not the discriminator; "
              "treat the verdict below as unproven.")

    # ---- verdict per cell -------------------------------------------------
    buckets = {"no skeleton": [], "stale root (filtered out)": [],
               "in selection, never reached": [], "attempted, produced nothing": []}
    for r in unembedded:
        if r in attempted:
            buckets["attempted, produced nothing"].append(r)
        elif r not in has_skel:
            buckets["no skeleton"].append(r)
        elif r not in latest:
            buckets["stale root (filtered out)"].append(r)
        else:
            buckets["in selection, never reached"].append(r)

    print("\n=== verdict ===")
    for name, ids in buckets.items():
        print(f"{name}: {len(ids)}")
        if ids:
            print(f"    by label: {dict(Counter(labels[r] for r in ids).most_common())}")
            for r in ids[:MAX_LISTED]:
                print(f"      {r}  {labels[r]}  n_skel={counts.get(r, 0)}")
            if len(ids) > MAX_LISTED:
                print(f"      ... and {len(ids) - MAX_LISTED} more")

    n_resumable = len(buckets["in selection, never reached"])
    print(
        f"\n{n_resumable} of {len(unembedded)} would be picked up by simply resubmitting "
        f"submit_embed_cells.sh; the rest need a different fix."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
