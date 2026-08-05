# The from-scratch local data pipeline is deprecated

**Deprecated 2026-08-04, replaced 2026-08-05.** `data/build_dataset.py`,
`data/cave_skeletons.py`, and `data/public_reader.py`'s raw-embedding-fetch-and-match path is
superseded by `data/build_dataset_from_store.py`, which reads directly from the real,
properly-indexed segclr-db store at `/orcd/compute/sdorkenw/001/collina/segclr-db` (see
"Replacement" below). The deprecated script's outputs were renamed out of the way so the new
pipeline could claim the paths `data/dataset.py` and every training script already expect:
`data/manifest_deprecated.json` + `data/graph_cache_deprecated/*.pt` (was
`data/manifest.json` + `data/graph_cache/`) hold the OLD data;
`data/manifest.json` + `data/graph_cache/` are now the NEW pipeline's output.
`data/skeleton_cache/*.pkl` is shared between both -- a Skeleton for a given root_id is the
same regardless of which pipeline fetched it. Test results already produced by the deprecated
pipeline are archived under `results/deprecated_local_pipeline/`, not deleted, but should not
be cited as performance numbers (see that directory's README for an additional overfitting
concern found independent of the embedding-correspondence problem). Kept in the repo as a
fallback (see "Why keep it around" below), and because `gnn/` and `baseline/` are data-source
agnostic — they consume whatever produces a manifest + `torch_geometric.data.Data` per cell,
so nothing there needed to change for the pivot.

One more thing worth flagging even though it doesn't affect the deprecation reasoning: the
replacement store's embeddings are from `resnet_860b_reshuffled`, a model this lab
trained/ran (datastack `minnie65_phase3_v1`, mat_version 1718) — **not** Google's public
SegCLR release from the original paper, which is what the deprecated pipeline used. The user
confirmed switching to this model is intentional, not an oversight.

## Why

This pipeline fetched public GCS SegCLR embeddings and CAVE skeletons **independently** and
reconciled them by nearest-neighbor xyz matching (`build_one_cell` in `build_dataset.py`),
because the public embedding release has no edges/topology at all — just
`node_id, x, y, z, embedding[64]` rows, and that `node_id` is Google's own internal,
unexported skeletonization index, unrelated to CAVE's. There is no shared identifier between
"an embedding" and "a CAVE skeleton node" in the public release, so any correspondence has to
be reconstructed spatially.

`scripts/check_match_quality.py` quantified how bad that reconstruction actually is, across
all 365 real cells (4,380,314 embedding-to-node matches), by comparing each match's residual
distance to the skeleton's own local process radius at that node — the right yardstick,
raised by the user, after an earlier comparison against overall cell span (>1mm) turned out
to be misleading reassurance:

```
median residual/radius ratio: 2.90
mean:                          5.64
fraction with residual > 1x local radius (landed outside the process):        85.9%
fraction with residual > 2x local radius (likely wrong branch/neurite):       66.7%
```

**85.9% of matches placed the embedding outside the process it was supposedly on**, and two
thirds landed at more than twice the local radius away — consistent with regularly jumping to
an adjacent branch or a neighboring neurite entirely, not just imprecise placement on the
right structure. This is not a minor approximation error; the `(root_id, node_id)`
correspondence this pipeline builds is not trustworthy as a graph structure for the GNN.

## Replacement

A real, populated segclr-db store exists at `/orcd/compute/sdorkenw/001/collina/segclr-db`
(dataset `microns`) with embeddings ingested directly against real skeleton backbones —
`node_embeddings/d64` (11.8M rows), `skeleton_nodes`/`skeleton_edges` (2,410 skeletons),
`agg_embeddings/d64` (35.4M rows, 3 registered `agg_spec`s), 58 checkpoints under 1 registered
experiment. Because it was ingested through segclr-db's normal write path, `(root_id,
node_id)` is a real foreign key by construction (per segclr_db's own design, see CLAUDE.md's
"segclr_db architecture" section) — no nearest-neighbor reconciliation needed.

**Fixed 2026-08-05**: as of 2026-08-04 this store was not actually readable — directories were
world-traversable and `.manifest` metadata files were world-readable (so
`SegCLRDatabase.tables()` row counts worked), but every actual data file
(`node_embeddings/d64.lance/data/*.lance`, `skeleton_nodes.lance/data/*.lance`, etc.) was
`-rw-------`, owned by `collina`, group `orcd_rg_fstor012_pi_sdorkenw` (`jcbliao` is not a
member). `collina` fixed this with a `chmod` to `o+r` (confirmed: files are now
`-rw-r--r--`; `jcbliao`'s group membership is unchanged, so it was the world-read route, not
a group-membership grant) — real reads now succeed. `scripts/explore_real_store.py` is the
script to (re-)run for a full inventory — it checks `tables()`, experiments,
cells/labels/splits, and does one real end-to-end cell read; it hit the permission error on
its first run (2026-08-04) and needs re-running now that access works.

Two things this store does NOT have, confirmed via `tables()`: `cells`/`cell_labels` are 0
rows and `splits` is 0 rows. So labels/splits come from elsewhere: CAVE's `cortical_neurons`
subset (`cell_type_multifeature_combo` + `proofreading_status_and_strategy`, filtered to
`status_axon=True`) at mat_version 1718 -- queried through the **public** `minnie65_public`
datastack rather than `minnie65_phase3_v1` (the store's own run metadata says
`minnie65_phase3_v1`, which needs CAVE "view" permission this account doesn't have; mat_version
1718 turned out to also be queryable through `minnie65_public`, which already works, sidestepping
the permission gap entirely -- confirmed via `scripts/check_cell_type_labels.py`, 2193/2193
labeled root_ids overlap with the store's cells). The split is constructed locally (reusing
`data.build_dataset.stratified_split`), not registered into the shared store.

## Why keep the deprecated code around

If the store's permissions can't be resolved in a reasonable time, this pipeline is the
fallback — it works, it's just approximate. If reused, that approximation should be treated
as a known, quantified limitation (the numbers above), not silently assumed away.
