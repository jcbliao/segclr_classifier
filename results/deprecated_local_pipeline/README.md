# Deprecated -- do not cite these numbers

Everything in this directory (`baseline_depth2.json`, `baseline_depth3.json`,
`gnn_scratch_depth2.json`, `gnn_scratch_depth3.json`, `pretrain_random/checkpoint_*.pt`) was
produced against `data/manifest.json` + `data/graph_cache/*.pt` from the **deprecated local
pipeline** — real CAVE skeletons (fetched directly from CAVE, not from a segclr-db store) and
real public SegCLR embeddings, joined by an approximate nearest-neighbor xyz match that turned
out to be unreliable (85.9% of matches land outside the local process radius). Full
explanation: `data/DEPRECATED.md`.

As of 2026-08-05 the real segclr-db store at `/orcd/compute/sdorkenw/001/collina/segclr-db` is
readable (permissions fixed), which has embeddings ingested directly against real skeleton
backbones with an exact `(root_id, node_id)` correspondence by construction. That's the source
for all results going forward — this directory is kept only as a record of what the pipeline
code produces, and specifically NOT as a performance claim about the GNN or the baseline.

One additional reason not to trust the GNN numbers here even setting the embedding-correspondence
problem aside: `gnn_scratch_depth3.json` hit 100% test accuracy, which on inspection looks like
overfitting on a small, class-imbalanced dataset (train_loss collapsed to ~0.0001 while
val_balanced_accuracy degraded mid-run, 0.993 -> 0.827 -> 0.660) combined with a partial
graph-size shortcut (glia vs. excitatory-neuron classes have non-overlapping node-count ranges),
not genuine superior use of embedding content. See conversation history / commit log for the
full node-count analysis.
