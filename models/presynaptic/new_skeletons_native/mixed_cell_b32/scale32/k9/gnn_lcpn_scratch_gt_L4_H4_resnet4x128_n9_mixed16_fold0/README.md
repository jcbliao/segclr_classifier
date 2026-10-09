# gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n9_mixed16_fold0

Experiment: `presynaptic/new_skeletons_native/mixed_cell_b32/scale32/k9`. Architecture: `graph_transformer`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.7389 | 0.7209 |
| Cell | 0.7863 | 0.8000 |

Source: [evaluation report](../../../../../../../results/presynaptic/new_skeletons_native/mixed_cell_b32/scale32/k9/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n9_mixed16_fold0.json). Values are copied without recomputation.

Classes: `putative_cge`, `putative_parvalbumin`, `putative_somatostatin`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_0`. Best epoch as recorded: `not recorded`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../docs/reproducibility.md).
