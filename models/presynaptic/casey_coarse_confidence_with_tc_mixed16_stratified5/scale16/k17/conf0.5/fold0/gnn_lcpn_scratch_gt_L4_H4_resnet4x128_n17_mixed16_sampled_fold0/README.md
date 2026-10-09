# gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0

Experiment: `presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.5/fold0`. Architecture: `graph_transformer`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.6867 | 0.6899 |
| Cell | 0.7875 | 0.7739 |

Source: [evaluation report](../../../../../../../../results/presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.5/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0.json). Values are copied without recomputation.

Classes: `BasketFam`, `BipFam`, `MartFam`, `NglFam`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_0`. Best epoch as recorded: `not recorded`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../docs/reproducibility.md).
