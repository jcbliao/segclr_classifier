# gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold4

Experiment: `presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold4/pre_post`. Architecture: `pointwise_mlp`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.7328 | 0.7193 |
| Cell | 0.7878 | 0.7604 |

Source: [evaluation report](../../../../../../../../../results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold4/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold4.json). Values are copied without recomputation.

Classes: `BasketFam`, `BipFam`, `MartFam`, `NglFam`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_4`. Best epoch as recorded: `21`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../docs/reproducibility.md).
