# The two synapse databases: what they are and how to run them

Two files, one row per synapse, covering every cell in `data/manifest.json`:

| file | our cell is | holds |
|---|---|---|
| `data/synapse_cache/presynaptic_sites.parquet` | **presynaptic** | every location where one of our cells synapses *onto* something else |
| `data/synapse_cache/postsynaptic_sites.parquet` | **postsynaptic** | every location where something synapses *onto* one of our cells, with the presynaptic partner's `root_id` |

Built by `data/build_synapses.py` from CAVE's `synapses_pni_2` at
**datastack `minnie65_public`, materialization 1718** — the same materialization
`data/manifest.json` records for its labels and embeddings.

## Polarity is stated, not inferred

CAVE's synapse table carries `pre_pt_root_id` and `post_pt_root_id` on every row,
assigned by the automated cleft detector. Nothing in this pipeline infers direction
from geometry. `outgoing` filters on `pre_pt_root_id`, `incoming` on `post_pt_root_id`
(`data/synapses.py::MODES`).

## Schema

Both files have **identical columns**, so they concatenate and compare without a
translation step. `cell_*` is always our cell's side of the synapse and `partner_*`
the other cell's — which means `partner_root_id` is the **postsynaptic** partner in
the presynaptic file and the **presynaptic** partner in the postsynaptic file.

| column | meaning |
|---|---|
| `cell_root_id` | the `data/manifest.json` cell this row belongs to |
| `mode` | `outgoing` (cell is presynaptic) / `incoming` (cell is postsynaptic) |
| `synapse_id` | CAVE's own synapse id, stable within a materialization |
| `partner_root_id` | the other cell; **0** where its supervoxel resolved to nothing |
| `cell_supervoxel_id`, `partner_supervoxel_id` | the durable, materialization-independent keys |
| `cleft_size` | CAVE's `size`: cleft segmentation voxel count |
| `cell_x_nm`, `cell_y_nm`, `cell_z_nm` | the synaptic point on **our** cell |
| `partner_x_nm`, `partner_y_nm`, `partner_z_nm` | the same point on the partner |
| `ctr_x_nm`, `ctr_y_nm`, `ctr_z_nm` | the cleft centroid — "where the synapse is", independent of side |

Rows are sorted by `cell_root_id`, so a per-cell read prunes on parquet row-group
statistics:

```python
import pandas as pd
pre = pd.read_parquet(
    "data/synapse_cache/presynaptic_sites.parquet",
    filters=[("cell_root_id", "==", 864691135271970725)],
)
```

## Reading it

```python
import duckdb
duckdb.sql("""
  select partner_root_id, count(*) n
  from 'data/synapse_cache/postsynaptic_sites.parquet'
  where cell_root_id = 864691135271970725
  group by 1 order by n desc limit 10
""")   -- this cell's strongest presynaptic inputs
```

`partner_root_id = 0` is kept rather than dropped: the synapse is still a real
location on our cell, and dropping those rows would silently shrink the *location*
database to serve the *partner* one. Filter it out yourself when the partner is what
matters.

## Running it

```bash
python -u data/build_synapses.py --dry-run          # the plan, querying nothing
python -u data/build_synapses.py --limit 20         # a pilot, to measure throughput

sbatch --array=0-3%4 scripts/sbatch/build_synapses.sh   # the real run
MERGE=1 sbatch scripts/sbatch/build_synapses.sh         # fuse the shards
sbatch scripts/sbatch/check_synapses.sh                 # validate what came out
```

Cells are queried in groups of 10 (`filter_in_dict` takes a list), so the sweep is
~230 requests per direction rather than ~2,300 against a service other labs share;
each rank also sleeps between calls, and the array is deliberately narrow. Output is
one parquet **part per (chunk, mode)**, written to a temp file and renamed — so a
preempted rank loses at most the chunk it was on, re-running resumes, and a
half-written part can never be mistaken for a finished one. `--merge` streams the
parts into the two databases and refuses to run if any part is missing.

## Three ways this goes wrong silently, and what catches each

**Positions come back in voxels.** `synapses_pni_2` is stored at (4, 4, 40) nm/voxel
and `query_table` returns that resolution unless `desired_resolution=[1, 1, 1]` is
passed — with no difference in the column names. Every query passes it, and
`data/synapses.py::_assert_nanometres` rejects a chunk whose coordinates sit below the
voxel ceiling.

**A truncated query looks like a small one.** The server caps rows per query and
announces truncation only through a logged `Warning` header. So every fetch asks for
`get_counts` first and refuses to return a frame whose length disagrees with that
count; a group of cells over the cap is split in half and recursed on.

**A materialization mismatch renames cells.** root_ids are materialization-scoped, so
querying synapses at a version other than the manifest's would return ids naming a
different snapshot of the proofreading — the failure hard constraint #8 is about. The
builder refuses to run when `--mat-version` disagrees with the manifest's
`label_mat_version`.

## Validation

`scripts/check_synapses.py` covers coverage, polarity (re-queried from CAVE by
`synapse_id`, which is the only check that can catch a swapped `cell`/`partner`
prefix), reciprocity (a synapse between two of *our* cells must appear in both files
with the roles exactly swapped), nanometre units and volume bounds, and per-cell row
counts against a fresh CAVE count.

## Not done here

There is **no join from a synapse to a skeleton node** of `data/graph_cache/*.pt`, and
therefore none to an embedding. Doing it needs a decision this build deliberately does
not make for you: nearest-node-in-space is wrong wherever two branches of the same cell
pass close together, which near a synapse is common. `cell_supervoxel_id` is the honest
key to build that join on.

## Why not `segclr_db.cave.query_synapse_partners`

That function deduplicates to one representative synapse per partner supervoxel,
because what it feeds is a per-partner embedding. Deduplicating deletes synaptic
*locations* on our own cell, which is exactly what these two databases are for, so
`data/synapses.py` queries the table directly instead.
