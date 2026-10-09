# gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold4

Experiment: `presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold4/pre_post`. Architecture: `graph_transformer`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.8978 | 0.8889 |
| Cell | 0.9120 | 0.8788 |

Source: [evaluation report](../../../../../../../../../results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold4/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold4.json). Values are copied without recomputation.

Classes: `BasketFam`, `Excitatory`, `nonBasketInhibitory`.

Fold as recorded: `fold_4`. Best epoch as recorded: `29`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../docs/reproducibility.md).
