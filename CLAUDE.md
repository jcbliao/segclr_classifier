# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Build a cell-type classifier for SegCLR embeddings that beats the current baseline —
**mean pooling of per-node embeddings within a geodesic context window** (Elabbady et al.,
*Nat Methods* 2023, https://www.nature.com/articles/s41592-023-02059-8).

The baseline (and the SegCLR paper) classifies **per point, from a small context window
around that point**, then majority-votes per-point predictions up into a cell-level answer —
it is not, and never was, a whole-cell classifier. The whole point of building a new
aggregation method is to improve classification for that same small-context-window regime, so
the GNN classifies per row too. Context window sizes are held fixed at the baseline's values
so the comparison is apples-to-apples: **only the aggregation changes**, from a geodesic mean
over the window to a GNN over the window's local subgraph — never from a per-point classifier
to a per-cell one.

Concretely, the unit of training and inference is a **local neighborhood subgraph around one
point, scoped to roughly the baseline's `window_nm`** — a real graph with real message-passing
structure, just much smaller than a whole cell's skeleton (and much larger than a single
isolated node, which would have no neighbors to pass a message from at all). Each window is
classified independently and majority-voted per cell. This also means the number of training
examples per epoch is on the same order as the baseline's row count, not the cell count: a
whole-cell graph with one prediction per cell is a different, smaller-sample task and not what
this project tests.

Two further principles, both load-bearing:

- **The pipeline's input is always raw, unaggregated embeddings**, regardless of which
  aggregation method is being tested. Aggregation is a component of the model, not something
  baked into the cached data.
- **The baseline and the GNN must see the same cells, the same splits, and the same label
  hierarchy.** They are two configurations of one pipeline (see below), not two pipelines.

## The model

`gnn/model.py::WindowClassifier` — one local window subgraph in, one hierarchical cell-type
prediction out. Trained end to end on the classification objective alone; there is no
pretraining stage and no reconstruction path.

`ModelConfig.architecture` picks the aggregation method, and it is the **only** thing that
differs between the two configurations. Same windows, same raw embeddings, same head, same
evaluation:

| `architecture` | Aggregation | Role |
|---|---|---|
| `graph_transformer` (default) | `gnn/graph_transformer.py::GraphTransformer` — AC-attention stack with an internal CLS token | the attention GNN |
| `mpnn` | `gnn/encoder.py::MPNNEncoder` (GraphSAGE, 2 layers by default, no attention) + `MeanReadout` | the plain message-passing GNN |
| `mean` | `gnn/readout.py::MeanReadout` over raw node embeddings, no encoder, zero parameters | the mean-pool baseline |

Read as a ladder of how much learned mixing happens before the readout: none, fixed local
neighbor averaging over a few hops, or adjacency-biased global attention. `MPNNEncoder` is
deliberately SAGE-only — attention is what `graph_transformer` is for, and keeping them
separate architectures rather than one `conv_type` flag is the point. Its default depth of 2
is tied to window size: windows average 10.7 nodes and `window_nm` is a radius, so deeper
stacks over-smooth a graph that small toward a constant vector.

**GraphTransformer** takes the attention mechanism and transformer stack of Weis et al.'s
GraphDINO (https://github.com/marissaweis/ssl_neuron, `ssl_neuron/graphdino.py`) — the
architecture only, not GraphDINO's self-distillation pretraining (no teacher/student pair, no
EMA, no centering/temperature, no projector head). Its bias mechanism: attention logits are a
per-node-predicted trade-off between global dot-product attention and a fixed adjacency bias,
`attn = gamma_0 * (QK^T/sqrt(d)) + gamma_1 * adj` (adjacency includes self-loops), with
`gamma = predict_gamma(x)` learned per node per layer and passed through `exp()` so both
weights stay positive.

Two adaptations beyond a literal port, both because this project's unit is a small
(~10-node-average) local window rather than a whole skeleton subsampled to a fixed node count:

1. **Padding, not subsampling.** Variable-size windows are padded per batch
   (`to_dense_batch`/`to_dense_adj`) with an explicit key-padding mask in `GraphAttention`,
   rather than forced to one fixed `n_nodes`. Without the mask, padded all-zero key positions
   could still receive softmax weight from the unconstrained global-attention term.
2. **The Laplacian positional encoding is precomputed per window**, not batched —
   `data/geodesic_window.py::_window_laplacian_pos_enc` runs once per window at extraction
   time (attached as `Data.pos_enc`, riding through PyG batching like `x`). Eigendecomposing
   one shared padded matrix across many different-size real windows would mix real
   eigenvectors with spurious ones from the zero-adjacency padding block.

Each node also carries `rel_pos`: its xyz minus the window's center node's xyz, so the center
sits at (0,0,0) and every other node at a translation-invariant offset. It is concatenated
onto `x` (not added like `pos_enc`) — a SegCLR embedding's context window is a function of its
xyz source position, but nothing in the embeddings themselves encodes *where within the window*
a node sits relative to the point being classified.

#### GraphTransformer ablation switches

Four independent switches, all defaulting to the full model, so each can be turned off without
touching the others:

| `ModelConfig` | CLI | Off means |
|---|---|---|
| `gt_use_lpe` | `--gt-no-lpe` | no additive Laplacian PE term (`to_pos_embedding` isn't built) |
| `gt_use_rel_pos` | `--gt-no-rel-pos` | the center-relative geometry isn't concatenated — dx, dy, dz **and their norm**, all 4 channels together |
| `gt_use_adj_bias` | `--gt-no-adj-bias` | plain scaled dot-product attention; `predict_gamma` isn't built |
| `gt_attention_scope` | `--gt-attention-scope neighborhood` | hard `-inf` mask restricting attention to 1-hop neighbors, instead of global attention with adjacency as a soft bias |

Two further switches are **on**-by-request rather than off-by-request:
`gt_use_dist_bias` / `--gt-dist-bias` (below) and `gt_use_thickness` / `--gt-use-thickness`
(see the dendrite-thickness section — parked, not recommended).

One further switch is **on**-by-request rather than off-by-request, because it needs an extra
ingested cache: `gt_use_thickness` / `--gt-use-thickness` concatenates the spine-corrected
dendrite shaft radius (+ a measured flag) onto the node features, the same way `rel_pos` is
added. The one flag turns on both the dataset and the model side so they cannot drift. Read the
coverage caveat in the dendrite-thickness section below before interpreting a run with it.

`rel_pos` is **4 channels**, not 3: dx, dy, dz and their norm `‖(dx,dy,dz)‖`. The norm is
derived inside the model rather than cached, so it costs no dataset rebuild and nothing on the
per-window hot path. It's handed over explicitly because a ReLU MLP approximates
`sqrt(dx²+dy²+dz²)` badly — a smooth convex function of three inputs, which a piecewise-linear
network can only tile with planes. Direction and distance are complementary: the components
keep orientation (apical trunks run pia-ward), the norm makes "how far out in the window"
directly legible. One switch controls all four.


#### Learned distance bias (`--gt-dist-bias`)

Replaces the binary `adj` inside the bias term with a learned per-head scalar indexed by
**binned edge length**: `attn = γ₀·QKᵀ/√d + γ₁·b_head[bucket(d_ij)]`. Off by default.

Why this rather than a fixed `exp(-d/σ)` or Dwivedi & Bresson-style edge features
(arXiv 2012.09699):

- Today's `adj` is binary, so **every neighbor of a node gets the identical boost**. Measured
  over 2.1M real edges, the within-window max/min edge-length ratio has median **4.73×** and
  p90 **17.9×** (`scripts/check_edge_length_distribution.py`) — that whole spread is currently
  invisible to attention.
- `predict_gamma` broadcasts one γ across all heads, so heads can't specialize local vs.
  global. A per-head bucket table is the only part of the bias term a single head can shape.
- It's a **strict generalization**: set every distance bucket equal and the no-edge bucket to
  0 and you recover binary adjacency exactly. That's the initialization, verified in
  `scripts/smoke_test_model.py` to reproduce the base model bit-for-bit (max diff 0.0).
- The shape is learned, not assumed — an exponential decay can't represent a non-monotonic
  preference, a bucket table can.
- D&B's mechanism is multiplicative, keeps a per-layer edge hidden state, and presupposes
  neighbor-restricted attention. It's built for rich categorical edge attributes (bond type);
  here the edge carries **one scalar**, and multiplicative gating has the wrong inductive bias
  for distance — at zero content similarity it has no effect at all, whereas additive bias
  shifts the logit regardless of content.

Buckets: 0 = no edge, 1 = self-loop, 2+ = `DIST_BIAS_BOUNDARIES_NM` bins (chosen from the
measured p5 529 / p50 1867 / p95 3865 / p99 6117 nm distribution). Requires `gt_use_adj_bias`
— without a bias term there's nothing for it to occupy, and that combination raises rather
than silently doing nothing.

**This is the only consumer of `edge_attr` anywhere in `gnn/`.** The binary-adjacency bias and
the MPNN's `SAGEConv` both discard it, so before this switch existed, edge lengths were
computed, cached, sliced per window and shipped to the GPU entirely unused.

Caveat worth holding: `edge_attr` is `‖coords[src] − coords[dst]‖`, and with `rel_pos` plus its
norm as node features, `‖a−b‖² = ‖a‖² − 2a·b + ‖b‖²` means attention could in principle
synthesize pairwise distance already. So this overlaps with a feature that measured inert, and
a null result would not be surprising.

A disabled switch **drops its parameters**, it does not merely skip them at runtime — so an
ablated run carries no dead weights for the optimizer to allocate state for, and the
`gt_params` counts in `scripts/smoke_test_model.py`'s output are a direct check that a switch
did what it claims. A disabled input-side switch also stops *requiring* its input, so an
ablated run can be driven without ever computing `pos_enc`/`rel_pos`.

Two things that would otherwise be silent traps, both handled in
`GraphTransformer._neighborhood_mask` and covered by the smoke test:

- Under neighborhood scope the **CLS row and column are forced fully open**. `adj_full` gives
  CLS a self-loop and nothing else, so a purely adjacency-derived mask would leave CLS
  attending only to itself — and since CLS's final state *is* the returned embedding, every
  window would collapse to the same constant vector.
- The mask **diagonal is forced True on every position, padding included**, so no query row is
  ever entirely `-inf`. An all-masked row comes out of softmax as NaN. The 1-node windows that
  really occur in this data are exactly the case that would trip this.

**Interaction worth knowing before reading an ablation grid:** under `neighborhood` scope the
adjacency bias is nearly inert. The hard mask already restricts each node row to exactly the
positions where `adj == 1`, so `gamma_1 * adj` adds the same constant to every surviving logit
and cancels in the softmax. What still differs is `gamma_0`, the learned per-node temperature
riding along with the bias term — so "neighborhood + bias" vs. "neighborhood, no bias" is a
temperature ablation, not a structure ablation. The structure comparison is against `global`.

Enabled ablations are appended to the run name (`_nolpe`, `_norelpos`, `_noadjbias`, `_nbhd`),
so an ablation never overwrites the full run's `epoch_metrics.csv`.

**Classification head is LCPN (local-classifier-per-node)** for both configurations, not a flat
softmax — that's what the lab's own trained models use. One small classifier per internal
branch point of a hierarchy tree (root: neuron vs. non_neuron; down to 24 granular Allen-style
+ glia classes), cascaded top-down at inference. Tree and mechanics (masked per-node CE loss,
top-down cascade) are ported from `segCLR_cell_classification`'s `LocalClassifierSNGPTrainer` /
`get_local_classifier_nodes`, with the SNGP machinery stripped out since nothing else in `gnn/`
uses it. See `gnn/hierarchy.py` (`LAB_HIERARCHY_TREE`, `parse_hierarchy`,
`get_local_classifier_nodes`) and `gnn/lcpn.py` (`LCPNHead`).

#### Classification head: linear probe or the lab's ResNet

By default the per-node heads sit directly on the readout embedding — a **linear probe**.
`--cls-resnet` inserts `gnn/resnet.py::DeepResNetTrunk` first, shared across all nodes:

| | |
|---|---|
| default | readout → per-node `Linear` |
| `--cls-resnet` | readout → shared ResNet trunk → per-node `Linear` |

That second form is the lab's own `local_classifier_resnet_sngp`, the model their production
LCPN config actually trains — a shared backbone with one head per hierarchy node, the trainer
routing to heads by per-node mask. The trunk is a faithful port of their
`src/models/resnet.py::DeepResNet` (pre-activation residual blocks, `norm → relu → linear`
twice then a skip connection), **minus SNGP**: theirs wraps every dense layer in
`spectral_norm` and swaps the output for a `RandomFeatureGP`, which this project has
consistently stripped since nothing here consumes the uncertainty estimates.

Defaults `--cls-resnet-hidden 128 --cls-resnet-layers 4` come from their
`configs/local_classifier_sngp.yaml`, not from the class signature (which predates it at 32).
BatchNorm is available but off, as theirs is — it would mix statistics across every window in
a batch.

**This is orthogonal to `architecture`** and composes with all three, so it is tagged
separately in the run name (`_resnet{layers}x{hidden}`) outside the aggregation tag —
otherwise a `--cls-resnet` run would overwrite its linear-probe counterpart.

Because each LCPN tree node has a different number of children, no single `(B, num_classes)`
logits tensor exists. `forward()` returns the readout embedding `g`; callers use
`model.cls_head.compute_loss(g, targets)` and `model.cls_head.predict_top_down(g)`.

**Class weighting is on by default.** Per-node inverse-frequency weights computed from
train-split *window* counts (`gnn/lcpn.py::compute_node_class_weights`,
`data/dataset_lcpn.py::train_window_counts_by_label`). Without it the imbalance (L4IT ~2.45M
windows vs. singleton classes) drives the model to just predict populous classes — balanced
accuracy sits near chance while raw accuracy looks fine. Under this imbalance, **always report
balanced accuracy, macro F1, and per-class recall alongside raw accuracy**; raw accuracy alone
is not a sufficient result.

### Evaluation

`scripts/train_gnn.py::evaluate()` returns window-level predictions plus each window's
`root_id`. Cell-level metrics come from majority-voting those up
(`gnn/metrics.py::majority_vote_by_group`) — the same two-stage design the baseline uses, and
the headline number. Window-level metrics are reported alongside as a diagnostic.

Checkpoint selection uses **window** balanced accuracy: cell metrics majority-vote only a few
hundred val cells and are genuinely noisy epoch to epoch, whereas window metrics average over
~1.8M windows and are what the training loss is directly shaped by.

## Data

`data/manifest.json` (labels + split) + `data/graph_cache/*.pt` (one `Data` per cell: raw
64-dim `resnet_860b_reshuffled` node embeddings, skeleton edges, xyz coords), built by
`data/build_dataset_from_store.py` reading segclr_db's store as a library.

- **Labels** come from segclr_db's own registered `cell_labels` table
  (`db.get_labels(label_set="cell_type")`), not a live CAVE query — so dataset builds don't
  depend on a `CAVE_TOKEN`.
- **Coverage: 2192 cells, 18 granular classes**, all under the `neuron` branch of
  `LAB_HIERARCHY_TREE`. `thalamocortical` and the four glia classes never receive gradient.
  This is an **embedding**-availability gap, not a labeling one: `cell_labels` does name
  non-neuron cells, but the `resnet_860b_reshuffled` experiment was scoped to the neuron-only
  subset and has zero embedding rows for them (`scripts/check_new_cells_embedding_coverage.py`).
  Switching label sources cannot fix it.
- **Chandelier cells (ChC) are excluded** (`EXCLUDED_LABELS`): n=1 in the store, too few to
  train or hold out on. `LAB_HIERARCHY_TREE`'s `putative_parvalbumin: [PV, ChC]` node is left
  structurally intact, so that branch is a permanent PV-only predictor.
- **Split is 80/20 train/test, per whole cell, with no separate val partition.**
  `data/build_dataset.py::stratified_split` assigns one split label per `root_id`, so no cell's
  nodes are ever split across partitions. "val" is an **alias for the test split** at the
  consumer level — `scripts/train_gnn.py` sets `val_ds = test_ds` (the same object, not a
  second load). Consequence, called out at the checkpoint-selection site: the best-epoch
  checkpoint is not held out from the final reported test metrics. Accepted trade-off, so that
  both partitions get the full 20% of held-out cells.

### Windows

`data/build_window_membership.py` precomputes, per cell, which nodes fall inside each node's
10µm geodesic window → `data/window_membership/*.npz`. Membership depends only on skeleton
structure, so it is unaffected by which embeddings are attached and does not need rebuilding
when the label source or split changes — only when the graph cache gains new cells. Average
window size is 10.7 nodes.

`data/geodesic_window.py::extract_window_subgraph` cuts one window on the fly (boolean-index
the window's nodes, remap edges, compute `pos_enc` and `rel_pos`, build a `Data`).
`data/dataset_windowed.py::WindowedGraphDatasetLCPN` is the PyG-facing loader: one item per
(cell, node) pair, with cell graphs and membership held resident in memory.
`data/dataset_lcpn.py::SegCLRGraphDatasetLCPN` is the whole-cell equivalent, for diagnostics.

### DataLoader throughput

`__getitem__` does real per-item CPU work, so `num_workers=0` serializes all ~8.7M window
extractions per epoch on one core with the GPU idle: ~30 min/epoch. **Raising `batch_size`
alone makes it worse** — batch size was never the bottleneck. `num_workers` (with
`persistent_workers=True`) is the fix: at `batch_size=4096`, 7 workers → ~6-7.6 batches/s,
15 → ~13-16, 31 → ~18.6-23.5 (clearly sublinear by then). Epoch time drops to ~1.5 min.

Keep `--num-workers` one below `--cpus-per-task`, leaving a core for the main process.
`scripts/sbatch/train_gnn.sh` requests **32 CPUs / 31 workers**. Note that on `mit_normal_gpu`
(not the default partition — see SLURM below) this account's QOS caps it at 32 CPUs total, so a
single 32-CPU job there blocks every other GPU job; splitting the budget across two concurrent
16-CPU jobs wins on total wall-clock for a side-by-side comparison sweep, precisely because
worker scaling is already sublinear past 15.

### Dendrite thickness

Per-vertex, spine-corrected dendrite shaft radius, computed by ray casting from a local mesh
patch: rays are cast from each skeleton vertex within the plane perpendicular to its local
skeleton tangent, and the median first-hit distance across the ring (with a few passes of
recentring) approaches the true shaft radius. Confining rays to that plane is what rejects
spines by construction — the radius CAVE skeletons already carry (`Skeleton.radii`) is measured
against the mesh with spines attached, so it reports "shaft plus spine heads" rather than the
shaft that actually conducts current. Ported from a private repo, `isebenius/E-I`'s
`src/dendrite_thickness.py` (fetched via `gh api`, per explicit user direction) — the estimator
itself (`data/dendrite_thickness.py`) is unchanged mesh+skeleton math; what's new for this
project is `skeleton_for_ray_casting`, bridging segclr_db's own `Skeleton` dataclass (already
carrying `compartments`/`radii` from CAVE, so no live skeleton fetch is needed) into the minimal
vertices/edges object the estimator actually reads.

Mesh fetching (`data/neuron_mesh.py::fetch_local_mesh_patch`) is deliberately a **local patch
only, never a whole-neuron mesh** — per explicit user direction ("each point likely only needs
the 3x3x3 L2 cache window from CAVE"). `cv.mesh.get(root_id, bounding_box=...)` restricts the
fetch server-side to the L2 fragments intersecting a small box around a batch of nearby
skeleton vertices, grouped into spatial buckets (`DEFAULT_BUCKET_SIZE_NM=80µm`,
`data/build_dendrite_thickness.py`) so several vertices share one fetch rather than one fetch
per vertex.

Output: `data/dendrite_thickness_cache/{root_id}.npz`'s `radius_nm`, aligned to
`skeleton.coords` (NaN where unmeasured — non-dendrite, branch point, or a mesh-hole miss).
Complete for all 2192 cells. See **`data/DENDRITE_THICKNESS.md`** for how to run it.

**Wired into the model as an opt-in node feature**, `--gt-use-thickness` (off by default; see
the GraphTransformer switch table above). The join is by `orig_node_ids` — a real id-based
index back into the skeleton vertex array, never a coordinate match. The feature is **two
channels, not one**: normalized radius, plus a *measured* flag. The flag isn't optional
bookkeeping — NaN is the cache's normal value for every axon node, every branch point, and
every mesh-hole miss, so a single channel would either poison the batch with NaN or claim
"radius 0" for a third of the graph.

> **Parked: not used, and not recommended.** The flag stays wired and runnable, but
> `gt_use_thickness` defaults off and no run should turn it on without a reason — per explicit
> user direction, after the coverage analysis below. Don't re-litigate this by default.
>
> **The measurement itself is sound.** Of the 3.63M nodes where a dendrite shaft radius is even
> defined, **100.0% were measured** — the ray-cast failure rate is 0.03% (1,189 nodes out of
> 11.79M). Low coverage is not failure.
>
> **But only 30.8% of nodes are eligible at all**, because the estimator measures dendrite
> shaft at non-branch-point vertices, and **68.7% of nodes are axon** (0.5% branch points,
> 0.02% soma). "Shaft radius" is undefined on an axon, so those are refusals by design.
>
> **That makes the measured flag effectively a dendrite-vs-axon indicator, which separates E/I
> on its own.** Per-node measured share runs 16.4% (NMC) / 16.5% (PV) / 20.2% (MC) at one end to
> 51.5% (L6CT) / 57.7% (L5ET) at the other, and the measured and axon columns sum to ~99% for
> every class — so the spread is skeleton composition (interneuron reconstructions are dominated
> by dense local axon), real morphology rather than a mesh artifact. It is still a class signal
> reachable without reading a single radius, which is why any gain from this feature would need
> a **mask-only control run** before being called a result about thickness.
>
> Regenerate all of this with `scripts/check_thickness_features.py` (join + per-class coverage)
> and `scripts/check_thickness_coverage_breakdown.py` (decomposes unmeasured into
> axon / branch point / bad tangent / real failure, from cached skeletons only — no mesh fetch).

Measured radii themselves look biologically sane — median shaft radius ~250–310 nm per cell,
p5–p95 roughly 140–490 nm.

### Deprecated data paths

`data/build_dataset.py`, `data/public_reader.py`, `data/cave_skeletons.py` implement an earlier
pipeline that joined public-release embeddings onto CAVE skeletons by xyz nearest-neighbor.
That reconciliation is quantifiably unreliable (85.9% of matches land outside the local process
radius) — see **`data/DEPRECATED.md`**. The modules stay because `stratified_split`,
`label_vocab`, and the skeleton pickle cache are still used, but do not build new datasets
through them. Results produced by that pipeline have been deleted from `results/`; don't cite
any that turn up elsewhere as performance numbers.

## Repository layout

```
gnn_classifier/
  gnn/                   the model
    model.py               WindowClassifier + ModelConfig -- picks the aggregation method
    graph_transformer.py   AC-attention GraphTransformer (encoder + CLS readout, fused)
    encoder.py             MPNNEncoder -- plain GraphSAGE message passing, no attention
    readout.py             MeanReadout -- pooling for the mpnn and mean architectures
    lcpn.py                LCPNHead: per-node heads, masked CE loss, top-down cascade
    resnet.py              DeepResNetTrunk -- ported from the lab, optional shared backbone
                             under LCPNHead in place of the linear probe
    hierarchy.py           LAB_HIERARCHY_TREE + parse_hierarchy/get_local_classifier_nodes
    metrics.py             summarize(), majority_vote_by_group()

  data/                  the data layer
    build_dataset_from_store.py  segclr_db store -> manifest.json + graph_cache/*.pt
    build_window_membership.py   graph_cache -> window_membership/*.npz
    geodesic_window.py           window_membership() + extract_window_subgraph()
    dataset_windowed.py          WindowedGraphDatasetLCPN (per-window; what training uses)
    dataset_lcpn.py              manifest/hierarchy loading + whole-cell dataset
    dataset.py, build_dataset.py, public_reader.py, cave_skeletons.py   (deprecated, above)
    dendrite_thickness.py, neuron_mesh.py, build_dendrite_thickness.py  (ray-cast shaft
      radius ingestion -- see DENDRITE_THICKNESS.md)

  scripts/               entry points; scripts/sbatch/ has the matching .sh for each
    train_gnn.py           training + evaluation (--architecture graph_transformer | mean)
    smoke_test_model.py    synthetic forward/backward pass over both architectures
    smoke_test_geodesic_window.py
    norm_diagnostic.py, norm_only_classifier.py    embedding diagnostics
    check_*.py, explore_*.py                       one-off data/CAVE diagnostics

  analysis/              training_curves.ipynb -- reads results/<run>/epoch_metrics.csv
  results/               per-run checkpoint_{best,last}.pt + epoch_metrics.csv, plus a
                         <run>.json summary alongside the directory. Holds 9 runs: meanpool,
                         mpnn_L2, gt_L4_H4 and its ablations (nolpe, norelpos, noadjbias,
                         nbhd, distbias, distbias_resnet4x128). Everything predating the
                         current model (4-channel rel_pos, the mpnn architecture) was cleared
                         deliberately as no longer comparable, so these are the only citable
                         numbers -- don't cite figures from a conversation log, read the CSV.
                         Epoch counts differ per run (100 down to 16); a preempted run keeps
                         its best checkpoint and every completed epoch, so a short CSV is
                         truncated, not a converged run. _archive_10epoch/ holds an earlier
                         10-epoch distbias_resnet4x128 before it was rerun longer.
                         Only the .pt files are gitignored; metrics are committed.
  logs/                  SLURM job output

  segclr_db/             git clone of dorkenwald-lab/segclr_db (branch main, upstream intact).
                         Used as a LIBRARY: aggregate, cave, skeletons, and the read side of
                         SegCLRDatabase. Its Writer/registry path is not used by this project.
  segCLR_cell_classification/   clone of the lab's own training codebase (reference)
```

Prefer adding new code in a sibling directory under `gnn_classifier/` over editing
`segclr_db/src/`, so the clone stays pullable. If a change to `segclr_db` is genuinely needed,
say so explicitly rather than committing into someone else's repo silently.

Run names encode the aggregation method so runs don't collide: `gnn_lcpn_scratch_meanpool`,
`gnn_lcpn_scratch_mpnn_L{layers}`, `gnn_lcpn_scratch_gt_L{depth}_H{heads}`.

## Hard constraints on how work gets done

These are user requirements, not preferences:

1. **Never execute project code on the Claude terminal's node** — not on a login node and not
   on a compute node, even when the shell happens to be on one (`hostname` here often returns
   a `node####`, which is *not* permission to run there). No `python train.py`, no `pytest`,
   no interactive Python. This includes "quick" one-liners that import numpy/lance/torch.
   Reading files, grepping, and inspecting the git tree are fine.
2. **All execution goes through a batch script submitted with `sbatch`.** Write the `.sh`,
   submit it, then poll `squeue`/read the log.
3. **`uv` is the environment manager.** Not conda, not bare `pip`. `uv` is at `~/.local/bin/uv`.
   The upstream `segclr_db` docs say `module load miniforge && source activate segclr` —
   **ignore that**; it is the lab's convention, not this project's.
4. **All compute — not just project code — goes through `sbatch`, including package installs.**
   The Claude terminal runs directly on a login node (confirmed: `hostname` → `login010`,
   `$SLURM_JOB_ID` empty) inside an apptainer container with no job allocation. `uv pip install`
   of anything heavy (torch, cloudvolume, compiled extensions) loads the shared login node
   exactly like running project code would — wrap it in a `.sh` and `sbatch` it
   (`scripts/sbatch/setup_env.sh`, `install_cloudvolume.sh` are the pattern). Lightweight
   metadata ops remain fine directly: `ls`/`cat`, `squeue`/`sacct`, `sbatch` submission itself,
   `chmod`, `jq` on small local files, `git`, reading/grepping repo files.
5. **All training/eval/inference runs on GPU nodes** — even for models small enough that CPU
   would technically work. Requesting a GPU partition isn't sufficient by itself:
   the script must actually call `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
   and move both model and tensors onto it, or the GPU allocation sits unused while everything
   silently runs on CPU anyway.
6. **Every training loop uses a `tqdm` progress bar, not periodic `print`s.** An outer
   epoch-level bar with `set_postfix(...)` updated *every* epoch, plus an inner per-batch bar
   (`leave=False`). Milestone lines use `tqdm.write(...)` (not `print`, which corrupts an active
   bar) so the log keeps permanent greppable per-epoch records alongside the live bar. Also run
   training scripts with `python -u` in the sbatch wrapper — without it Python block-buffers
   stdout when not a TTY, and a live job can sit silently for many minutes.
7. **H200 is the default GPU for GraphTransformer training.** Measured GPU utilization on
   those runs is high, so they are GPU-bound and a faster card pays. `scripts/sbatch/train_gnn.sh`
   requests `--gres=gpu:h200:1` explicitly rather than a generic `--gres=gpu:1`, which can
   schedule onto the scarce H100 pool; SLURM has no "prefer one type, fall back to another"
   within a single submission, so the explicit request queues for that type only.

   **The `mean` and `mpnn` architectures are not GPU-bound and gain nothing from this.**
   Measured on L40S: `meanpool` — a zero-parameter aggregator — ran at 175 s/epoch while
   `mpnn_L2` with two SAGEConv layers ran at 131 s/epoch. A model doing essentially no GPU work
   is not faster, so both sit at a data-pipeline floor of ~130–175 s/epoch. Override those to
   `gpu:l40s:1` (far more numerous, shorter queue) rather than competing for H200s. For
   reference on L40S, GraphTransformer runs cost ~340–354 s/epoch and `--gt-dist-bias` ~444.
   Check availability with `sinfo -p mit_preemptable -o "%N %G %t"`.
8. **Objects with different CAVE root IDs must NEVER be matched to each other.** A `root_id` is
   a real foreign key — two rows only refer to the same physical cell if their `root_id`s are
   identical, full stop. Never substitute a spatial/heuristic match (nearest-neighbor
   coordinates, name similarity, "close enough") across two different root_ids as a stand-in
   for a failed id-based lookup. Root IDs drift as CAVE proofreading continues (merges/splits
   change chunkedgraph IDs), so two different root_ids can legitimately be two snapshots of
   what a human calls "the same cell" — that does not make them interchangeable without an
   explicit, verified id-reconciliation step. When ids don't line up, track down why or abandon
   that data source; never snap the nearest points anyway.

## Environment

`uv` venv at `segclr_db/.venv` (CPython 3.11), with `segclr-db` installed editable.
There is **no `uv.lock`** and no `[tool.uv]` block — the venv was built with `uv pip`, so keep
using `uv pip` rather than `uv sync`/`uv add`.

```bash
cd segclr_db
uv venv --python 3.11                       # if recreating
uv pip install -e ".[all]"                  # read path + cave + aggregate + graph
uv pip install -e ".[dev]"                  # adds pytest + ruff (NOT currently installed)
```

Installed: pylance 9.0.0, duckdb 1.5.5, pyarrow 25, numpy 2.4, pandas 2.3, caveclient 8.2.1,
numba 0.66, networkx 3.6, h5py — plus, for the GNN side (all via `sbatch`, see hard
constraint #4): **torch 2.6.0+cu124**, **torch-geometric 2.8.0.post1**, **scikit-learn 1.9.0**,
**gcsfs**, **scipy**, **cloudvolume 12.14.4**.

Python is only on `PATH` inside the venv (`segclr_db/.venv/bin/python`); there is no system
`python`. Batch scripts must reference the venv interpreter by absolute path or activate it.

`cloudvolume` is a gotcha worth flagging: it's needed by `caveclient`'s
`SkeletonClient.generate_bulk_skeletons_async` (which `segclr_db.cave.CAVESkeletonSource.
_request_generation` calls), but segclr_db's `cave` extra declares only `caveclient`. Without
it, generation fails with `ImportError: Could not import cloudvolume` — and because
`_request_generation` wraps that in a broad `try/except` logged only at `debug` level, the
failure is **completely silent**: the readiness-poll loop waits forever for skeletons that were
never queued. If skeleton fetching plateaus with no progress for several minutes, check this
before assuming it's CAVE latency.

**CAVE token**: not `CAVE_TOKEN` by default in this environment — read it from
`~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json` (`jq -r .token ...`) and export
it in the sbatch script. Never print the token value into logs or output.

## SLURM

Account is `mit_general`. Partitions and walltime caps:

| Partition | Limit | Use for |
|---|---|---|
| `mit_preemptable` | 2-00:00:00 | **default for anything needing a GPU** — training, GPU smoke tests, long CAVE ingests |
| `mit_quicktest` | 15:00 | **default for short CPU-only jobs** — diagnostics, import checks. **No GPUs** |
| `mit_normal` | 12:00:00 | CPU work too slow for quicktest's 15-min cap (aggregation, data prep, network-bound checks) |
| `mit_normal_gpu` | 6:00:00 | GPU work that must not be preempted, or when preemptable's GPU quota is full |

**`mit_quicktest` has no GPUs at all** — `GRES` is `(null)` on all 26 nodes (`sinfo -p
mit_quicktest -o "%N %G"`), so a `--gres=gpu:...` request there simply never schedules. Checked,
not assumed. Hence the split above: a GPU smoke test goes to `mit_preemptable` (it runs in
~11 s, so preemption risk is irrelevant), and only CPU-only work can use quicktest.

Pick quicktest for a CPU job whenever it plausibly finishes in 15 minutes — the repo's
diagnostics mostly run in under a minute (`check_thickness_features` 55 s,
`check_thickness_coverage_breakdown` 52 s over all 2192 cells). Reserve `mit_normal` for jobs
that are genuinely long or network-bound against CAVE, where the 15-min cap is a real risk.

**`mit_preemptable` is the default partition for GPU jobs**, with `--cpus-per-task=32` (and
therefore `--num-workers 31`). It has the longest walltime cap and generally the shortest
queue; the cost is that jobs can be preempted. `scripts/train_gnn.py` is partially resilient to
that — `checkpoint_best.pt` is written the moment a new best epoch appears and
`epoch_metrics.csv` is appended per epoch, so a preempted run keeps its best weights and every
completed epoch's metrics — but **there is no resume-from-checkpoint**, so a resubmitted run
restarts from epoch 0.

Per-user QOS caps differ by partition and are separate pools — check with
`sacctmgr show qos -P format=Name,MaxTRESPerUser`:

| Partition | Per-user cap |
|---|---|
| `mit_preemptable` | cpu=1024, gpu=4, mem=4T |
| `mit_normal_gpu` | cpu=32, gpu=2, mem=515G |

That separation matters in practice: a large multi-node ingest on `mit_preemptable` (e.g. an
`embed_cells` job holding many L40S) will park a training job behind it with reason
`QOSMaxGRESPerUser`, while `mit_normal_gpu`'s own 2-GPU allowance stays free. Check
`squeue -u $USER` before assuming a pending job is just queue depth, and override the partition
on the command line (`sbatch --partition=mit_normal_gpu scripts/sbatch/train_gnn.sh`) rather
than editing the script when it's a one-off. `mit_normal_gpu` is also the choice for a run that
genuinely must not be interrupted, keeping its 6h cap in mind.

Skeleton ingestion is the other long pole: CAVE's skeleton service allows ~10 requests/minute,
so a few thousand cells takes hours and the full set takes days. Use `mit_preemptable` with
`examples/ingest_skeletons.py`, which is resumable by design.

Writable space: `/orcd/home/002/jcbliao/...` and `/orcd/scratch/orcd/013/jcbliao`. Note
`/orcd/home/002/jcbliao/rotation/...` and `/home/jcbliao/rotation/...` are the **same
directory** (identical device+inode); the editable install records the `/home/jcbliao` form.

`/orcd/compute/sdorkenw/001/collina/segclr-db` is the shared store and the `DEFAULT_DB_ROOT`
baked into `schema.py` — the one every `SegCLRDatabase()`/`SegCLRWriter()` built without an
explicit `root=` will silently use. It is readable now, but pass `root=` explicitly anyway
rather than relying on the default.

### Scale characteristics that affect job design

Per-cell node counts are long-tailed: p50 532, p90 3.1k, p99 8.8k, max 20.5k. 38% of cells have
under 100 nodes but 0.6% of nodes; the largest 1% hold 11%. Consequences for SLURM array jobs:
split tasks by **cumulative node count, not cell count**, and give each task a **contiguous**
`root_id` block rather than a strided slice, so each commit covers a narrow range and Lance
fragment statistics prune on read. Concurrent appenders are fine — Lance optimistic-concurrency
retry handles it.

## Commands

**Everything that executes Python must be wrapped in an sbatch script** — written plainly here
for reference, not for direct invocation.

```bash
# training (via scripts/sbatch/train_gnn.sh; ARCHITECTURE / EPOCHS / NUM_WORKERS env vars)
python -u scripts/train_gnn.py                       # GraphTransformer (default)
python -u scripts/train_gnn.py --architecture mpnn   # 2-layer GraphSAGE + mean readout
python -u scripts/train_gnn.py --architecture mean   # mean-pool baseline
python -u scripts/train_gnn.py --gt-no-lpe           # -> results/gnn_lcpn_scratch_gt_L4_H4_nolpe/
python -u scripts/train_gnn.py --gt-attention-scope neighborhood
python -u scripts/train_gnn.py --gt-dist-bias        # -> ..._gt_L4_H4_distbias
python -u scripts/train_gnn.py --cls-resnet          # -> ..._gt_L4_H4_resnet4x128
python -u scripts/train_gnn.py --architecture mean --cls-resnet   # head choice is orthogonal

# segclr_db's own test suite, run from segclr_db/
pytest                              # all tests; real Lance store in a tmpdir, no CAVE token
pytest -m "not slow"                # skip the multi-process concurrency test
pytest -m "not integration"         # skip anything needing a live CAVE token / network
ruff check src tests examples && ruff format --check src tests examples   # line-length 100
```

## segclr_db reference (the vendored data layer)

Read `segclr_db/USAGE.md` for the practical API tour and `segclr_db/DESIGN.md` for why.
`segclr_db/stubs.py` is an implementation-free interface listing — the fastest way to see the
whole surface at once. `examples/quickstart.py` is the end-to-end tour and doubles as
executable documentation.

Storage is Lance, queried with DuckDB, laid out as
`<root>/<dataset>/{registry,dims,skeletons,embeddings,predictions}/`. Datasets (`microns`,
`v1dd`, `h01`) are independent stores; no cross-dataset queries.

Load-bearing ideas:

- **Reads go through `SegCLRDatabase`, writes through `SegCLRWriter`** — separate classes so a
  read-only user never sees a write method. `store.py` is the only module importing `lance` or
  `duckdb`; keep it that way.
- **An experiment is a config.** `experiment_id` is unique, guarded by a hash over an
  *experiment spec* (a projection of the config that the **training repo** owns, not the
  database). A run is `<experiment_id>__<timestamp>`; a checkpoint is `checkpoint_e{E}_s{S}`.
  `latest`/`best` resolve at registration and are never stored, so every row traces to specific
  weights. A differing spec raises `ExperimentConfigDrift` rather than quietly redefining the
  name — put operational keys in `spec_exclude`, scientific keys in the spec.
- **`(root_id, node_id)` is a real foreign key.** `node_id` is the index into the CAVE
  skeleton's `vertices` array and is the *only* node identifier in the system. Coordinates live
  on `skeleton_nodes` alone and are joined on request (`return_coords=True`), never duplicated
  onto embedding rows.
- **Embedding dim selects a physical table** (`node_embeddings/d64`, `/d128`), because Lance's
  `fixed_size_list` is fixed-width. `embedding_dim` lives on the experiment, so callers never
  pass or see `D`. A query spanning experiments of different `D` raises `MixedEmbeddingDimError`.
  In `raw_sql`, tables appear with the suffix (`node_embeddings_d64`).
- **Completion is a table.** `work_units` records every attempt with status `ok` | `empty` (ran,
  produced nothing) | `error` (retried) | `refused` (permanent, never retried). A cell that never
  ran has no row. This is why *deleting files is not how you redo work* — use
  `writer.drop(..., dry_run=True)` first, which removes data and its work records together.
- **Registration is the chokepoint.** Ingestion never auto-creates registry rows; a typo'd
  `experiment_id` raises `UnregisteredError` listing the known ones. Amending a spec is refused
  once embeddings exist — register a new experiment instead.
- **`add_embeddings` is a primitive** with no already-written check. In ingest loops use
  `add_cell_embeddings`, which is per-cell and therefore resumable, and which validates
  `node_id` against the cached skeleton.
- **`SCHEMA_VERSION` (currently 2) is bumped by hand** in `schema.py` whenever a stored value's
  shape *or meaning* changes. Opening an older store fails outright; there is no in-place
  migration by design — re-create and re-ingest.
- **`root_ids=` accepts a bare int or a list.**

Classifier runs live in the **same registry** with `kind="classifier"` (`EXPERIMENT_KINDS =
("segclr", "classifier")`). The `predictions` / `prediction_logits` / `prediction_runs` tables
are **specified in the schema but not yet implemented**, so writing GNN predictions back will
mean implementing that path.

### Aggregation — the baseline's definition

`src/aggregate.py::geodesic_mean` computes, per node, the mean of every embedding within
`window_nm` **geodesic** distance along the skeleton (not Euclidean — two branches passing close
in space do not contribute to each other). The kernel is a bounded Dijkstra per source node
compiled with numba, ~3.2 µs/node flat regardless of cell size. `window_nm = 0` is identity,
which is how "raw" is expressed without a special case.

For the GNN, the graph is `skeleton_edges` — stored as raw directed `(root_id, src, dst)` pairs
that **readers symmetrize**. `aggregate.build_csr(skeleton)` gives CSR `(offsets, neighbors,
weights)` with edge lengths in nm, closer to what a PyG `edge_index` wants than
`Skeleton.to_networkx()` (which needs the `[graph]` extra). Node coordinates come from
`skeleton_nodes` via `get_embeddings(return_coords=True)` or `Skeleton.coords` — never from the
embedding rows.

### Splits and labels

Splits are data (`writer.create_split(...)` / `db.get_split(...)`), not a side effect of Dataset
construction — a cell in two splits is refused, since that leakage is the failure this prevents.

Label hierarchy levels are derived at read time (`db.get_labels(hierarchy_id=...)`), never
materialized. A label the hierarchy doesn't cover comes back with **null levels rather than
being dropped**, so check for nulls — otherwise your dataset silently shrinks.

## Notion

**Spec** — "Graph AutoEncoder Classifier" (page id `3b2d7b0592128092b307fef34d65d0b4`) is the
original design page. Note the model has since diverged from it substantially: the
masked-autoencoder pretraining, the message-passing encoder, and the CLS-query attention
readout it describes are all gone. Treat "The model" section above as authoritative for what
exists; the page is history.

**Progress log** — "Claude Updates by Day"
(https://app.notion.com/p/Claude-Updates-by-Day-3b3d7b059212801db512e3bdc797c35a, page id
`3b3d7b059212801db512e3bdc797c35a`). **Whenever the user says "update notion", populate this
page.** `ntn pages edit` **replaces** page content rather than appending, so a real update is
fetch-then-append:

```bash
ntn pages get 3b3d7b059212801db512e3bdc797c35a   # current content
# append a new dated entry to what came back, then:
ntn pages edit 3b3d7b059212801db512e3bdc797c35a --content '<old content>\n\n## <date>\n\n<new entry>'
```

Write a concise, dated entry (what changed, what was validated, what's blocked/decided) — not a
transcript dump. Plain `WebFetch` on these URLs redirect-loops, so `ntn` is the only way in. If
`ntn doctor` reports "no token found", the user must authenticate — `ntn login` is interactive,
so suggest `! ntn login` rather than running it yourself.

## Related work

A fellow lab member's notebook, `analysis/presynaptic-distributions.ipynb` on the `main` branch
of https://github.mit.edu/collina/segCLR_cell_classification (also has `aggregation_study`,
`cleanup`, `hierarchical`, `lcpn`, `refactor`, `simple_classifiers` branches) — a pointer only;
the user will add real documentation on it in their own style. SSH access confirmed working
(`git@github.mit.edu`).
