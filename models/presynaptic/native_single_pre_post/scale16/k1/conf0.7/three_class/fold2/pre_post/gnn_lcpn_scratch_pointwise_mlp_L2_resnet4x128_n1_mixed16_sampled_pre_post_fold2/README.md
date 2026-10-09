# gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold2

Experiment: `presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold2/pre_post`. Architecture: `pointwise_mlp`.

## Saved evaluation

| Scope | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Window | 0.8183 | 0.8169 |
| Cell | 0.9243 | 0.9010 |

Source: [evaluation report](../../../../../../../../../../results/presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold2/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold2.json). Values are copied without recomputation.

Classes: `BasketFam`, `Excitatory`, `nonBasketInhibitory`.

Fold as recorded: `fold_2`. Best epoch as recorded: `29`.

## Configuration and artifacts

- [Recorded training arguments](config.json)
- [Full evaluation metadata and per-class metrics](metrics.json)
- [Checkpoint publication metadata](artifact.json)

The artifact metadata records verified checkpoint publication details; null fields are unverified.

Training arguments are historical records. Dataset paths must be supplied for your environment; older reports may omit options introduced later. See [reproduction guidance](../../../../../../../../../../docs/reproducibility.md).
