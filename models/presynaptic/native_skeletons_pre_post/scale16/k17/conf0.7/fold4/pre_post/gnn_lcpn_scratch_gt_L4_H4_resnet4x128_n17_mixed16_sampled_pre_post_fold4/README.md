# gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold4

Experiment: `presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold4/pre_post`. Architecture: `graph_transformer`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.7476 | 0.7301 |
| Cell | 0.8714 | 0.8213 |

Source: [evaluation report](../../../../../../../../../results/presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold4/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold4.json). Values are copied without recomputation.

Classes: `BasketFam`, `BipFam`, `MartFam`, `NglFam`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_4`. Best epoch as recorded: `28`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../docs/reproducibility.md).
