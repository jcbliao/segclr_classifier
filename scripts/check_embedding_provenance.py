"""Which job wrote the embeddings now in the store -- and specifically, did the
user's own embed_cells run write the 124 glia/thalamocortical cells?

`work_units.worker` is "{hostname}/{SLURM_PROCID}" (segclr_db/src/work.py::
worker_id), and finished_ts is per cell, so a row can be attributed to a job by
node + time window without guessing. The user's embed_cells jobs, from sacct:

    19884933  Aug 7 16:47:36 - 16:47:38   node[3501-3504]  (died at once)
    19885032  Aug 7 16:50:03 - 16:50:13   node[3501-3504]  (died at once)
    19885400  Aug 7 16:53:04 - 17:01:53   node[3501-3504]
    19886373  Aug 7 17:04:08 - Aug 8 04:06:03  node[3501-3504]   <- the 11h run
    19929333  Aug 8 08:42:29 - 08:46:37   node[4302,4308,4503-4504]  (cancelled)

Anything outside those windows was written by someone else's job (the store is
shared), which is the alternative hypothesis worth ruling out.

Run via sbatch (mit_normal). Read-only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

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

# Node AND time, both required. The store is shared and another embed job ran
# concurrently on other nodes, so a time window alone would credit its rows to
# the user's job wherever the two overlap.
NODES_A = {"node3501", "node3502", "node3503", "node3504"}
NODES_B = {"node4302", "node4308", "node4503", "node4504"}
JOBS = [
    ("19884933", "2026-08-07T16:47:36", "2026-08-07T16:47:38", NODES_A),
    ("19885032", "2026-08-07T16:50:03", "2026-08-07T16:50:13", NODES_A),
    ("19885400", "2026-08-07T16:53:04", "2026-08-07T17:01:53", NODES_A),
    ("19886373", "2026-08-07T17:04:08", "2026-08-08T04:06:03", NODES_A),
    ("19929333", "2026-08-08T08:42:29", "2026-08-08T08:46:37", NODES_B),
]

OTHER = "OTHER (someone else's job)"


def attribute(ts: pd.Timestamp, host: str) -> str:
    for job, start, end, nodes in JOBS:
        if host in nodes and pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return job
    return OTHER


def main() -> int:
    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)

    wu = st.scan(store, "work_units", filter="table = 'node_embeddings'").to_pandas()
    wu["host"] = wu["worker"].str.split("/").str[0]
    wu["job"] = [attribute(ts, h) for ts, h in zip(wu["finished_ts"], wu["host"], strict=True)]
    print(f"=== all {len(wu):,} node_embeddings work_units ===")
    print(f"finished_ts spans {wu['finished_ts'].min()} .. {wu['finished_ts'].max()}")
    print("\nby attributed job:")
    print(wu["job"].value_counts().to_string())
    print("\nby worker host (user's nodes marked):")
    for host, n in wu["host"].value_counts().items():
        tag = " <- user's job" if host in NODES_A | NODES_B else ""
        print(f"  {host} {n}{tag}")
    print("\nOTHER rows, time span (is a foreign job still running?):")
    oth = wu[wu["job"] == OTHER]
    print(f"  {len(oth):,} rows, {oth['finished_ts'].min()} .. {oth['finished_ts'].max()}")
    print(f"\ndistinct scope_id (run/checkpoint): {sorted(wu['scope_id'].unique())}")

    # ---- the 124 -----------------------------------------------------------
    ours = {int(rid) for rid in json.loads(MANIFEST_PATH.read_text())["cells"]}
    df = db.get_labels(label_set=LABEL_SET)
    labels = {int(r.root_id): str(r.label) for r in df.itertuples()
              if str(r.label) not in EXCLUDED_LABELS}
    embedded = {int(r) for r in db.cells_with_embeddings(EXPERIMENT_ID).tolist()}
    the_124 = sorted((set(labels) & embedded) - ours)

    sub = wu[wu["root_id"].astype("int64").isin(the_124)]
    print(f"\n=== the {len(the_124)} glia/thalamocortical cells ===")
    print(f"{len(sub)} work_unit rows")
    print("by attributed job:")
    print(sub["job"].value_counts().to_string())
    print("by worker host:")
    print(sub["worker"].str.split("/").str[0].value_counts().to_string())
    print(f"finished_ts spans {sub['finished_ts'].min()} .. {sub['finished_ts'].max()}")
    print(f"status: {dict(Counter(sub['status']))}")
    print(f"n_rows (embedded nodes) total={int(sub['n_rows'].sum()):,} "
          f"min={int(sub['n_rows'].min())} max={int(sub['n_rows'].max())}")
    print("\nfirst 10 rows:")
    cols = ["root_id", "status", "n_rows", "worker", "started_ts", "finished_ts", "job"]
    print(sub[cols].head(10).to_string(index=False))

    # ---- our 2192, for contrast -------------------------------------------
    mine = wu[wu["root_id"].astype("int64").isin(ours)]
    print(f"\n=== our 2192 manifest cells, for contrast ===")
    print(f"{len(mine)} work_unit rows")
    print(mine["job"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
