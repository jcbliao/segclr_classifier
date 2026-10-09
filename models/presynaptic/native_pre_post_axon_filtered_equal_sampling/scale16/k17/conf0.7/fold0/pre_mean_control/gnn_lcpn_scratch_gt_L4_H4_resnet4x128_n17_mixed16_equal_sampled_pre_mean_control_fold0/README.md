# gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0

Experiment: `presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_mean_control`. Architecture: `graph_transformer`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.6803 | 0.6920 |
| Cell | 0.8623 | 0.8182 |

Source: [evaluation report](../../../../../../../../../results/presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0.json). Values are copied without recomputation.

Classes: `BasketFam`, `BipFam`, `MartFam`, `NglFam`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_0`. Best epoch as recorded: `25`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../docs/reproducibility.md).
