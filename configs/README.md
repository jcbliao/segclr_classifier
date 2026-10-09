# Training configurations

These JSON files are recipes for the current trainer, not claims to reproduce any particular historical checkpoint. Keys match the trainer's argparse destination names (for example, `gt_no_lpe`, not `--gt-no-lpe`). Omitted keys use the current CLI defaults.

| Recipe | Aggregator |
|---|---|
| [baselines/mean.json](baselines/mean.json) | Raw mean pooling |
| [baselines/linear.json](baselines/linear.json) | Linear projection followed by mean |
| [baselines/pointwise_mlp.json](baselines/pointwise_mlp.json) | Two-layer pointwise MLP |
| [baselines/mpnn.json](baselines/mpnn.json) | Two-layer GraphSAGE on skeleton edges |
| [baselines/mpnn_complete.json](baselines/mpnn_complete.json) | Two-layer GraphSAGE on complete window graphs |
| [graph_transformer/fixed_node.json](graph_transformer/fixed_node.json) | Two-layer, four-head graph transformer |
| [ablations/graph_transformer_embeddings_only.json](ablations/graph_transformer_embeddings_only.json) | Graph transformer with LPE and relative-position inputs disabled; adjacency bias retained |

```bash
sbatch scripts/sbatch/train_config.sh configs/baselines/mean.json --num-embeddings 10 --results-dir results/reproductions/mean_n10
```

Explicit CLI flags take precedence over JSON. Unknown keys and invalid choices are rejected. Paths in JSON are interpreted relative to the training working directory, which the lab launcher sets to the repository root. For a fresh reproduction, choose a separate `--results-dir`; the new launcher requires one to avoid overwriting existing runs.

The [model catalog](../models/README.md) separately stores the actual arguments copied from each saved final report. Historical argument records can omit newer options, so inspect the original launcher and dataset metadata before replaying them.
