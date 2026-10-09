# gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold0

Experiment: `presynaptic/cave_embedding_augmentation_conf0.7`. Architecture: `graph_transformer`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.6582 | 0.6677 |
| Cell | 0.8137 | 0.7931 |

Source: [evaluation report](../../../../results/presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold0.json). Values are copied without recomputation.

Classes: `BasketFam`, `BipFam`, `MartFam`, `NglFam`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_0`. Best epoch as recorded: `23`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../docs/reproducibility.md).
