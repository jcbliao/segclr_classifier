# gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold1

Experiment: `presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold1/pre_post`. Architecture: `pointwise_mlp`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.8152 | 0.8125 |
| Cell | 0.9127 | 0.8922 |

Source: [evaluation report](../../../../../../../../../../results/presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold1/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold1.json). Values are copied without recomputation.

Classes: `BasketFam`, `Excitatory`, `nonBasketInhibitory`.

Fold as recorded: `fold_1`. Best epoch as recorded: `24`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../../docs/reproducibility.md).
