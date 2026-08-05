# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Build a cell-type classifier for SegCLR embeddings that beats the current baseline —
**mean pooling of per-node embeddings within a geodesic context window** (Elabbady et al.,
*Nat Methods* 2023, https://www.nature.com/articles/s41592-023-02059-8). The hypothesis is
that a **GNN over the skeleton graph** does better than the mean. Context window sizes are
held fixed at the baseline's values so the comparison is apples-to-apples; only the
aggregation/readout changes.

That framing determines what the baseline is in code: `agg_embeddings` rows produced by
`geodesic_mean` at a registered `window_nm` (see Aggregation below). A GNN experiment
consumes the **raw** `node_embeddings` plus `skeleton_edges` for the same cells, and must be
evaluated against the *same* cells, splits, and label hierarchy as the mean-pool baseline.

> **Validation-phase deviation, decided explicitly by the user:** the first validation uses
> segclr_db **only as a library** (`aggregate.geodesic_mean`/`build_csr`, `cave.CAVESkeletonSource`,
> `skeletons.normalize_cave_skeleton`, the `Skeleton` dataclass) — never `SegCLRDatabase` /
> `SegCLRWriter` / the Lance store. Real public SegCLR embeddings, real CAVE skeletons, and
> real ground-truth labels are ingested and cached locally (`gnn_classifier/data/`) instead of
> into a registered segclr-db experiment. This keeps the "same cells/splits/aggregation
> definition" guarantee above without the registration ceremony. See "Public SegCLR data
> source (validation phase)" below for exactly what was validated and how.
>
> **This local pipeline is now DEPRECATED (2026-08-04)** — the nearest-neighbor xyz
> reconciliation it uses to join embeddings onto CAVE skeleton nodes is quantifiably
> unreliable (85.9% of matches land outside the local process radius). A real, properly-indexed
> segclr-db store at `/orcd/compute/sdorkenw/001/collina/segclr-db` replaces it — **readable as
> of 2026-08-05** (was permission-blocked; `collina` fixed it with `chmod o+r`). Past test
> results from the deprecated pipeline are archived under `results/deprecated_local_pipeline/`,
> not to be cited as performance numbers. Full detail: **`data/DEPRECATED.md`**.

The chosen architecture is a **masked-autoencoder-pretrained graph encoder with a CLS-query
attention readout** — see "Planned model" below. It lives in a Notion page, "Graph AutoEncoder
Classifier", which is the source of truth and may have been updated since this file was
written. Read it with the `ntn` CLI:

```bash
ntn pages get 3b2d7b0592128092b307fef34d65d0b4
```

Plain `WebFetch` on the URL redirect-loops, so `ntn` is the only way in. If `ntn doctor`
reports "no token found", the user must authenticate — `ntn login` is interactive, so
suggest `! ntn login` rather than running it yourself.

## Planned model (from the Notion page)

The rationale for a graph at all: a SegCLR embedding's context window is a function of its
xyz source position, so a skeleton graph over embeddings should expose features of that
context window that a mean discards.

Notation: skeleton `G = (V, E)`, `N = |V|`, embeddings `X ∈ R^(N×D)`, coordinates
`P ∈ R^(N×3)`, adjacency `A`, masked node set `M ⊆ V`.

**1. Masked input.** Select 30% of nodes. Each selected node `i` is replaced with a learned
mask embedding `m` w.p. 0.8, a replacement embedding `x_r` drawn from another skeleton w.p.
0.1, and left as `x_i` w.p. 0.1. The three sampling choices for `x_r` are all worth trying:
a random other neuron, a random neuron of a *different* class, a random neuron of the *same*
class. Resample the corruption pattern every epoch. **Never mask the CLS query.**

**2. Encoder.** `Z = f_θ(X̃, A)`, `Z ∈ R^(N×d)`.

**3. Decoder.** `x̂_i = d_φ(z_i)`, `x̂_i ∈ R^D`.

**4. Pretraining objective.** Masked node-feature reconstruction, computed over **all
selected nodes `M`** — including the 10% whose inputs were left unchanged — and **only**
over `M`. Do not compute the primary reconstruction loss on unmasked nodes.

Cosine loss is the natural first choice for SegCLR embeddings:
`L_cos = (1/|M|) Σ_{i∈M} (1 − ⟨x̂_i, x_i⟩ / (‖x̂_i‖‖x_i‖ + ε))`.
If magnitude turns out to carry information, add Smooth L1:
`L_mask = L_cos + λ_mag · (1/|M|) Σ_{i∈M} SmoothL1(x̂_i, x_i)`.
Which one to use is **decided by a norm diagnostic, not guessed** — see Diagnostics below.
The choice is between normalized SegCLR targets and cosine loss; only test the combined
loss if the diagnostic says norms matter.

**5. Readout.** Start with a **CLS query** `q_CLS` that reads node embeddings but does not
write back to the skeleton: `g = Attention(q_CLS, Z, Z)`. Single-head form:
`α_i = softmax_i((W_q q_CLS)^T W_k z_i / √d_k)`, `g = Σ_i α_i W_v z_i`. A later variant can
send `g` back into the graph for global communication. Hierarchical pooling is deliberately
**not** the starting point — the page's stated skepticism is that stacking layers and then
mean-pooling is unlikely to beat just using the embeddings, which is the very baseline this
project is trying to beat.

**6. Classification head.** `ŷ = softmax(W g + b)`. Compare frozen encoder vs fine-tuned
encoder vs joint objective.

`L_cls = −(1/B) Σ_s log p(y_s | G_s)`, switching to class-weighted
`−(1/B) Σ_s w_{y_s} log p(y_s | G_s)` if classes are substantially imbalanced. **Weights are
selected from training data only.** Under imbalance, report balanced accuracy, macro F1, or
per-class recall alongside raw accuracy — raw accuracy alone is not a sufficient result.

Joint training masks part of the graph for the reconstruction path while leaving enough
signal for classification: `L_joint = L_cls + λ_rec · L_mask`.

### Diagnostics the page asks for

Run an **embedding-norm diagnostic**: (1) how variable are the norms, (2) do they predict
anything, (3) can they be ablated (`scripts/norm_diagnostic.py`).

> **Decided explicitly by the user:** cosine loss is the principled default for SegCLR
> embeddings and pretraining starts with it **regardless of what the diagnostic finds** — the
> diagnostic no longer gates the cosine-vs-cosine+SmoothL1 choice, it is informative only. It
> would matter for a *later* decision about whether to add the SmoothL1 magnitude term, not
> for the starting choice. `use_smooth_l1=False` is the default everywhere
> (`gnn/losses.py`, `scripts/pretrain_gnn.py`).

## Progress log (Notion)

"Claude Updates by Day" — https://app.notion.com/p/Claude-Updates-by-Day-3b3d7b059212801db512e3bdc797c35a
(page id `3b3d7b059212801db512e3bdc797c35a`) is the running progress log for this project,
separate from the "Graph AutoEncoder Classifier" spec page above.

**Whenever the user says "update notion" (this project's session, not a general instruction),
populate this page with progress using the `ntn` CLI.** `ntn pages edit <page-id> --content
'...'` **replaces** the page's content rather than appending — so a real update is
fetch-then-append, not a direct write:

```bash
ntn pages get 3b3d7b059212801db512e3bdc797c35a   # current content
# append a new dated entry to what came back, then:
ntn pages edit 3b3d7b059212801db512e3bdc797c35a --content '<old content>\n\n## <date>\n\n<new entry>'
```

Write a concise, dated entry (what changed, what was validated, what's blocked/decided) —
not a full transcript dump. Same `ntn doctor`/`ntn login` auth caveats as the spec page: if
`ntn doctor` reports no token, suggest `! ntn login` rather than running it yourself.

## Hard constraints on how work gets done

These are user requirements, not preferences:

1. **Never execute project code on the Claude terminal's node** — not on a login node and
   not on a compute node, even when the shell happens to be on one (`hostname` here often
   returns a `node####`, which is *not* permission to run there). No `python train.py`, no
   `pytest`, no `examples/quickstart.py`, no interactive Python. This includes
   "quick" one-liners that import numpy/lance/torch.
2. **All execution goes through a batch script submitted with `sbatch`.** Write the `.sh`,
   submit it, then poll `squeue`/read the log. Reading files, grepping, and inspecting the
   git tree are fine — those are not running project code.
3. **`uv` is the environment manager.** Not conda, not bare `pip`. `uv` is at
   `~/.local/bin/uv`. Note the upstream `segclr_db` docs say
   `module load miniforge && source activate segclr` — **ignore that**; it is the lab's
   convention, not this project's.
4. **All compute — not just project code — goes through `sbatch`, including package
   installs.** The Claude terminal's interactive session runs directly on a login node
   (confirmed: `hostname` → `login010`, `$SLURM_JOB_ID` empty) inside an apptainer container
   with no job allocation. `uv pip install` of anything heavy (torch, cloudvolume, compiled
   extensions) run there loads the shared login node exactly like running project code would.
   Wrap installs in a `.sh` and `sbatch` them too (`scripts/sbatch/setup_env.sh`,
   `install_cloudvolume.sh` are the pattern). Lightweight metadata ops remain fine directly:
   `ls`/`cat`, `squeue`/`sacct`, `sbatch` submission itself, `chmod`, `jq` on small local
   files, `git`, reading/grepping repo files.
5. **All training/eval/inference runs on GPU nodes (`mit_normal_gpu`)** — even for models
   small enough that CPU would technically work (e.g. the mean-pool baseline). Requesting the
   partition isn't sufficient by itself: the script must actually call
   `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and move both model and
   tensors onto it, or the GPU allocation sits unused while everything silently runs on CPU
   anyway — this bit us once with `train_baseline.py`/`smoke_test_model.py` before both were
   fixed to move things onto the device properly.

## Environment

`uv` venv at `segclr_db/.venv` (CPython 3.11), with `segclr-db` installed editable.
There is **no `uv.lock`** and no `[tool.uv]` block — the venv was built with
`uv pip`, so keep using `uv pip` rather than `uv sync`/`uv add`.

```bash
cd segclr_db
uv venv --python 3.11                       # if recreating
uv pip install -e ".[all]"                  # read path + cave + aggregate + graph
uv pip install -e ".[dev]"                  # adds pytest + ruff (NOT currently installed)
```

Installed: pylance 9.0.0, duckdb 1.5.5, pyarrow 25, numpy 2.4, pandas 2.3, caveclient 8.2.1,
numba 0.66, networkx 3.6, h5py — plus, for the GNN side (all via `sbatch`, see hard
constraint #4): **torch 2.6.0+cu124**, **torch-geometric 2.8.0.post1**, **scikit-learn
1.9.0**, **gcsfs**, **scipy**, **cloudvolume 12.14.4**.

`cloudvolume` is a gotcha worth flagging: it's needed by
`caveclient`'s `SkeletonClient.generate_bulk_skeletons_async` (which
`segclr_db.cave.CAVESkeletonSource._request_generation` calls), but segclr_db's `cave` extra
in `pyproject.toml` only declares `caveclient` — not this transitive dependency. Without it,
generation requests fail with `ImportError: Could not import cloudvolume`, and because
`_request_generation` wraps that call in a broad `try/except` logged only at `debug` level,
the failure is **completely silent**: the readiness-poll loop just waits forever for
skeletons that were never actually queued. If skeleton fetching plateaus with no progress for
several minutes, check for this before assuming the wait is legitimate CAVE latency.

Python is only on `PATH` inside the venv (`segclr_db/.venv/bin/python`); there is no system
`python`. Batch scripts must reference the venv interpreter by absolute path or activate it.

## SLURM

Account is `mit_general`. Partitions and walltime caps:

| Partition | Limit | Use for |
|---|---|---|
| `mit_normal` | 12:00:00 | CPU work (aggregation, data prep) |
| `mit_normal_gpu` | 6:00:00 | GNN training |
| `mit_quicktest` | 15:00 | smoke tests, import checks |
| `mit_preemptable` | 2-00:00:00 | long CAVE skeleton ingests, checkpoint-and-resume jobs |

Skeleton ingestion is the long pole: CAVE's skeleton service allows ~10 requests/minute, so
a few thousand cells takes hours and the full set takes days. Use `mit_preemptable` with
`examples/ingest_skeletons.py`, which is resumable by design.

`/orcd/compute/sdorkenw/001/collina/segclr-db` — the `DEFAULT_DB_ROOT` baked into
`schema.py`, and the store every `SegCLRDatabase()`/`SegCLRWriter()` built without an
explicit `root=` will silently use — **is not readable by this user** (`Permission denied`,
group is `sched_mit_hill`). Any code path that defaults to it will fail on the first call.
Pass `root=` explicitly, or resolve the access question with the user before designing
around the shared store. Writable space: `/orcd/home/002/jcbliao/...` and
`/orcd/scratch/orcd/013/jcbliao`.

Note `/orcd/home/002/jcbliao/rotation/...` and `/home/jcbliao/rotation/...` are the **same
directory** (identical device+inode); the editable install records the `/home/jcbliao` form.
Don't treat them as two trees.

## Repository layout

```
gnn_classifier/
  segclr_db/            git clone of dorkenwald-lab/segclr_db (branch main, upstream remote intact)
                         -- vendored dependency, used as a LIBRARY only (see above); its own
                         Store/Writer/Database/registry is not used by this project yet.

  data/                  the whole data layer for the validation phase, no segclr_db store:
    public_reader.py       vendored EmbeddingReader/md5_shard (Apache-2.0, ~100 lines lifted
                            from google-research/connectomics/segclr/reader.py -- NOT a pip
                            install of the full connectomics package, which pulls in
                            tensorflow/edward2 for the unrelated SNGP classifier submodule)
                            + wrappers for the label feather file and per-cell embeddings
    cave_skeletons.py       direct CAVE fetch (segclr_db.cave.CAVESkeletonSource, chunked with
                            incremental disk caching -- see chunk-size comment for why) +
                            local pickle cache, data/skeleton_cache/*.pkl
    build_dataset.py        labels + skeletons + embeddings -> one torch_geometric Data per
                            cell (data/graph_cache/*.pt) + data/manifest.json (stratified
                            train/val/test split). Threaded (ThreadPoolExecutor) for the
                            embedding downloads -- I/O-bound, so threads beat a SLURM array
                            here; skeleton fetch stays single-threaded since CAVE's rate limit
                            is shared regardless of caller count. Resumable.
    dataset.py              SegCLRGraphDataset (PyG-facing loader) + ReplacementPool (x_r
                            sampling for masking, drawn from train split only)

  gnn/                    the Graph AutoEncoder Classifier model (see "Planned model" above)
    masking.py, encoder.py, decoder.py, readout.py, model.py, losses.py, metrics.py

  baseline/
    mean_pool_classifier.py  geodesic-mean baseline via segclr_db.aggregate.geodesic_mean
                              (pure function, not the store) on the SAME cached cells/splits
                              the GNN uses

  scripts/                 entry points; scripts/sbatch/ has the matching .sh for each
    build_dataset.py, norm_diagnostic.py, train_baseline.py, pretrain_gnn.py, finetune_gnn.py,
    smoke_test_model.py, smoke_test_pretrain.py (real training scripts against synthetic data
    in an isolated data/_smoke_dataset/ path -- never touches real data/manifest.json or
    data/graph_cache/), explore_*.py / check_*.py (one-off data/CAVE diagnostics)
```

Prefer adding new code in a sibling directory under `gnn_classifier/` over editing
`segclr_db/src/`, so the clone stays pullable. If a change to `segclr_db` is genuinely
needed, say so explicitly rather than committing into someone else's repo silently.

## segclr_db architecture (the data layer you will build on)

Read `segclr_db/USAGE.md` for the practical API tour and `segclr_db/DESIGN.md` for why.
`segclr_db/stubs.py` is an implementation-free interface listing — the fastest way to see
the whole surface at once. `examples/quickstart.py` is the end-to-end tour and every
USAGE.md snippet is lifted from it, so it doubles as executable documentation (submit it via
sbatch if you want to see it run).

Storage is Lance, queried with DuckDB, laid out as `<root>/<dataset>/{registry,dims,skeletons,embeddings,predictions}/`.
Datasets (`microns`, `v1dd`, `h01`) are independent stores; no cross-dataset queries.

### Public SegCLR data source (validation phase) — validated facts, not guesses

Everything below was confirmed against the real services, not assumed — see
`scripts/explore_public_labels.py` and `scripts/explore_cave_alignment.py`.

- **Embeddings**: public, anonymous-GCS release from Elabbady et al. Read via a vendored
  `EmbeddingReader` (`data/public_reader.py`), keyed by dataset "data_key". Use
  `microns_nm_coord_public_offset_v343` for raw per-node embeddings — this is the variant
  whose xyz lines up with CAVE skeleton coordinates (both nm, public-offset frame). Plain
  `microns_v343` is in the internal segmentation's own voxel/coordinate frame and will **not**
  line up with CAVE — confirmed by trying both against a CAVE skeleton and comparing ranges.
  **Embedding dim is 64** (confirmed by inspecting a real fetched row, not assumed from the
  `d64`/`d128` example in segclr_db's own docs).
- **Ground-truth labels**: `gs://iarpa_microns/minnie/minnie65/embedding_classification/training_data/labeled_cell_m343_df_221011b.feather`
  — 398 labeled cells, columns `seg_id, pt_position, cell_type, status_dendrite, status_axon,
  clean_axon_only, status_whole`. `cell_type` is a dash-separated hierarchy string, coarse to
  fine (e.g. `C-N-I-BC-Pvalb` = cell / neuron / inhibitory / basket cell / Pvalb+; `C-G-OGC` =
  cell / glia / oligodendrocyte) — 20 distinct values, several with under 10 examples.
  `data/dataset.py::label_vocab(depth=...)` truncates to any hierarchy depth.
- **CAVE datastack**: `minnie65_public`, materialization **343** (matches the label table's
  `m343` and the embedding bucket's `_v343`). Note 343 is **no longer in the datastack's live
  `get_versions()` list** (materializations get retired) but skeleton fetching works against
  it anyway — skeletons key off root_id + skeleton_version, not a still-valid materialization.
  All 398 labeled root_ids were confirmed **latest** (`is_latest_roots`, none stale) —
  staleness was a real hypothesis raised and directly checked, and ruled out, when skeleton
  generation looked stuck (see the cloudvolume gotcha above, which was the actual cause).
- **CAVE token**: not `CAVE_TOKEN` env var by default in this environment — read it from
  `~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json` (`jq -r .token ...`) and
  export it in the sbatch script. Never print the token value itself into logs/output.
- **Coordinate-frame alignment**: nearest-neighbor distance from embedding xyz to the nearest
  CAVE skeleton node, over one real ~15k-row cell: median 740nm, mean 893nm, p95 2192nm, max
  7206nm — small relative to a cell spanning >1mm, consistent with "same structure, two
  different skeletonization algorithms," not a coordinate bug. `data/build_dataset.py` does
  this matching (`cKDTree`) for every cell and records `match_dist_median_nm` per cell as a
  sanity-checkable field.
- **Node-count mismatch is expected**: SegCLR's own per-node sampling is denser than CAVE's
  skeleton nodes (one real cell: 15,871 embedding rows vs. 7,082 skeleton nodes) — multiple
  embedding rows commonly land on one skeleton node; `build_one_cell` means over collisions.

Five load-bearing ideas:

- **Reads go through `SegCLRDatabase`, writes through `SegCLRWriter`** — separate classes so
  a read-only user never sees a write method. `store.py` is the only module importing
  `lance` or `duckdb`; keep it that way.
- **An experiment is a config.** `experiment_id` is unique, guarded by a hash over an
  *experiment spec* (a projection of the config that the **training repo** owns, not the
  database). A run is `<experiment_id>__<timestamp>` (double underscore); a checkpoint is
  `checkpoint_e{E}_s{S}`. `latest`/`best` resolve at registration and are never stored, so
  every row traces to specific weights. A differing spec raises `ExperimentConfigDrift`
  rather than quietly redefining the name — put operational keys in `spec_exclude`,
  scientific keys in the spec.
- **`(root_id, node_id)` is a real foreign key.** `node_id` is the index into the CAVE
  skeleton's `vertices` array and is the *only* node identifier in the system. Coordinates
  live on `skeleton_nodes` alone and are joined on request (`return_coords=True`), never
  duplicated onto embedding rows.
- **Embedding dim selects a physical table** (`node_embeddings/d64`, `/d128`), because
  Lance's `fixed_size_list` is fixed-width. `embedding_dim` lives on the experiment, so
  callers never pass or see `D`. A query spanning experiments of different `D` raises
  `MixedEmbeddingDimError`. In `raw_sql`, tables appear with the suffix
  (`node_embeddings_d64`).
- **Completion is a table.** `work_units` records every attempt with status
  `ok` | `empty` (ran, produced nothing) | `error` (retried) | `refused` (permanent, never
  retried). A cell that never ran has no row. This is why *deleting files is not how you
  redo work* — use `writer.drop(..., dry_run=True)` first, which removes data and its work
  records together.

Classifier runs live in the **same registry** with `kind="classifier"` (`EXPERIMENT_KINDS =
("segclr", "classifier")`) — a GNN classifier is an experiment with a config and
checkpoints, not a second mechanism. The `predictions` / `prediction_logits` /
`prediction_runs` tables are **specified in the schema but not yet implemented** (README
"Status"), so writing GNN predictions back will mean implementing that path.

### Aggregation — the baseline to beat

`src/aggregate.py` is the mean-pool baseline: `geodesic_mean` computes, per node, the mean
of every embedding within `window_nm` **geodesic** distance along the skeleton (not
Euclidean — two branches passing close in space do not contribute to each other). The kernel
is a bounded Dijkstra per source node compiled with numba, ~3.2 µs/node flat regardless of
cell size. `window_nm = 0` is identity, which is how "raw" is expressed without a special
case. `METHODS = {"geodesic_mean": geodesic_mean}` is the registry — a new aggregation
method is a new entry there plus a registered `agg_spec`.

A window is a **registered spec**, not a filename fragment:
`writer.register_agg_spec(window_nm=10000)` → `"geodesic_mean_10000nm"`. Re-aggregating
costs no SegCLR inference; it reads stored raw embeddings. Aggregation **requires a cached
skeleton** (geodesic distance needs the graph) and raises rather than silently skipping.

For the GNN, the graph is `skeleton_edges` — stored as raw directed `(root_id, src, dst)`
pairs that **readers symmetrize**. `Skeleton.to_networkx()` needs the `[graph]` extra;
`aggregate.build_csr(skeleton)` gives you CSR `(offsets, neighbors, weights)` with
edge lengths in nm, which is closer to what a PyG `edge_index` wants. The `P ∈ R^(N×3)`
coordinates the model spec lists come from `skeleton_nodes`, via
`get_embeddings(return_coords=True)` or `Skeleton.coords` — never from the embedding rows.

### Splits and labels

Splits are data (`writer.create_split(...)` / `db.get_split(...)`), not a side effect of
Dataset construction — a cell in two splits is refused, since that leakage is the failure
this prevents. Use these rather than re-deriving splits, so the GNN and the baseline train
on identical partitions.

Label hierarchy levels are derived at read time (`db.get_labels(hierarchy_id=...)`), never
materialized. A label the hierarchy doesn't cover comes back with **null levels rather than
being dropped**, so check for nulls — otherwise your dataset silently shrinks.

Two places the model spec touches this layer: the `x_r` replacement embeddings must be drawn
from the **training split only** (the spec says `r ~ D_train`), and the same-class /
different-class variants of that sampling need `get_labels` at the hierarchy level being
classified. Class weights for `L_cls` are likewise computed from the training split alone.

### Scale characteristics that affect job design

Per-cell node counts are long-tailed: p50 532, p90 3.1k, p99 8.8k, max 20.5k. 38% of cells
have under 100 nodes but 0.6% of nodes; the largest 1% hold 11%. Consequences for SLURM
array jobs: split tasks by **cumulative node count, not cell count**, and give each task a
**contiguous** `root_id` block rather than a strided slice, so each commit covers a narrow
range and Lance fragment statistics prune on read. Concurrent appenders are fine — Lance
optimistic-concurrency retry handles it.

## Commands

Run from `segclr_db/`. **Everything below that executes Python must be wrapped in an sbatch
script** — they are written plainly here for reference, not for direct invocation.

```bash
pytest                              # all tests; real Lance store in a tmpdir, no CAVE token
pytest -m "not slow"                # skip the multi-process concurrency test
pytest -m "not integration"         # skip anything needing a live CAVE token / network
pytest tests/test_aggregate.py::test_name -x    # single test
ruff check src tests examples && ruff format --check src tests examples   # line-length 100
```

Tests use a real Lance store in `tmp_path` (`tests/conftest.py`) rather than a fake backend
— a fake would only prove the fake works.

## Gotchas carried over from segclr_db

- **Registration is the chokepoint.** Ingestion never auto-creates registry rows; a typo'd
  `experiment_id` raises `UnregisteredError` listing the known ones.
- **Amending an experiment spec is refused once embeddings exist** — register a new
  experiment instead.
- **`add_embeddings` is a primitive** with no already-written check (a second call might be
  a duplicate or another shard, and it can't tell). In ingest loops use
  `add_cell_embeddings`, which is per-cell and therefore resumable, and which validates
  `node_id` against the cached skeleton.
- **`SCHEMA_VERSION` (currently 2) is bumped by hand** in `schema.py` whenever a stored
  value's shape *or meaning* changes. Opening an older store fails outright; there is no
  in-place migration by design — re-create and re-ingest.
- **`root_ids=` accepts a bare int or a list.**
