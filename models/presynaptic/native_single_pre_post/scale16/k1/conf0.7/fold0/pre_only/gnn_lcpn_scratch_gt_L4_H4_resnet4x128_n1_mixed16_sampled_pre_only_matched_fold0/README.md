# gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0

Experiment: `presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_only`. Architecture: `graph_transformer`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.5067 | 0.5012 |
| Cell | 0.5736 | 0.5719 |

Source: [evaluation report](../../../../../../../../../results/presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_only/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0.json). Values are copied without recomputation.

Classes: `BasketFam`, `BipFam`, `MartFam`, `NglFam`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_0`. Best epoch as recorded: `28`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../docs/reproducibility.md).
