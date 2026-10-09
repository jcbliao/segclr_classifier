# gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold1

Experiment: `presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold1/pre_post`. Architecture: `pointwise_mlp`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.7376 | 0.7325 |
| Cell | 0.9056 | 0.8738 |

Source: [evaluation report](../../../../../../../../../results/presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold1/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold1.json). Values are copied without recomputation.

Classes: `BasketFam`, `BipFam`, `MartFam`, `NglFam`, `pyramidal`, `thalamocortical`.

Fold as recorded: `fold_1`. Best epoch as recorded: `27`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../docs/reproducibility.md).
