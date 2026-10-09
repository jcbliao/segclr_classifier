# gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n40_fold0

Experiment: `all_windows`. Architecture: `pointwise_mlp`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.9009 | 0.9038 |
| Cell | 0.9735 | 0.9742 |

Source: [evaluation report](../../../results/all_windows/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n40_fold0.json). Values are copied without recomputation.

Classes: `astrocyte`, `microglia`, `oligo`, `putative_cge`, `putative_parvalbumin`, `putative_somatostatin`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_0`. Best epoch as recorded: `not recorded`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../docs/reproducibility.md).
