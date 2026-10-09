# Model catalog

Architectures are implemented in [`gnn/`](../gnn/). Each entry below is a trained run with a saved final evaluation report; an entry does not imply that its weights have been published.

Runs retain their experiment namespace. The table is sorted by path and makes no cross-experiment ranking.

[Machine-readable results](../results/summary.csv) · [Architecture reference](../docs/architectures.md) · [Reproduction](../docs/reproducibility.md)

Regenerate with `python scripts/build_model_catalog.py` (through Slurm on the lab cluster). Generated cards, configs, and metrics are overwritten; publication fields in `artifact.json` are preserved. Entries for removed reports are not automatically deleted.

193 final evaluation reports across 56 experiment directories.

## all_windows

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L2_H4_nolpe_norelpos_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_nolpe_norelpos_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_nolpe_norelpos_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_nolpe_norelpos_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_nolpe_norelpos_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_nolpe_norelpos_resnet4x128_n40_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_nolpe_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_nolpe_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_nolpe_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_nolpe_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_nolpe_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_nolpe_resnet4x128_n40_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_norelpos_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_norelpos_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_norelpos_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_norelpos_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_norelpos_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_norelpos_resnet4x128_n40_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L2_H4_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_gt_L2_H4_resnet4x128_n40_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_nolpe_norelpos_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_nolpe_norelpos_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_nolpe_norelpos_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_nolpe_norelpos_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_nolpe_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_nolpe_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_nolpe_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_nolpe_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_norelpos_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_norelpos_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_norelpos_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_norelpos_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n20_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mean_resnet4x128_n10_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_mean_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mean_resnet4x128_n20_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_mean_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mean_resnet4x128_n40_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_mpnn_L2_lpe_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_lpe_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_lpe_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_lpe_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_lpe_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_lpe_resnet4x128_n40_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_position_lpe_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_position_lpe_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_position_lpe_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_position_lpe_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_position_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_position_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_position_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_position_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_position_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_position_resnet4x128_n40_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L2_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mpnn_L2_resnet4x128_n40_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_lpe_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_lpe_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_lpe_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_lpe_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_lpe_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_lpe_resnet4x128_n40_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_position_lpe_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_position_lpe_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_position_lpe_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_position_lpe_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_position_lpe_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_position_lpe_resnet4x128_n40_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_position_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_position_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_position_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_position_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_position_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_position_resnet4x128_n40_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_resnet4x128_n10_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_resnet4x128_n20_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_mpnn_L4_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_mpnn_L4_resnet4x128_n40_fold0/README.md) | `mpnn` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_fold0/README.md) | `pointwise_mlp` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n20_fold0/README.md) | `pointwise_mlp` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n40_fold0](all_windows/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n40_fold0/README.md) | `pointwise_mlp` |
| [gnn_lcpn_scratch_pointwise_mlp_L4_resnet4x128_n10_fold0](all_windows/gnn_lcpn_scratch_pointwise_mlp_L4_resnet4x128_n10_fold0/README.md) | `pointwise_mlp` |
| [gnn_lcpn_scratch_pointwise_mlp_L4_resnet4x128_n20_fold0](all_windows/gnn_lcpn_scratch_pointwise_mlp_L4_resnet4x128_n20_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0/fold0/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.3/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.3/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.3/fold0/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.3/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.5/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.5/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.5/fold0/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.5/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.7/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.7/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.7/fold0/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.7/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.9/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.9/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.9/fold0/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0](presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf0.9/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_confidence_cave/k10/conf0/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0/fold0/gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_confidence_cave/k10/conf0.3/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.3/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.3/fold0/gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.3/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_confidence_cave/k10/conf0.5/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.5/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.5/fold0/gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.5/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_confidence_cave/k10/conf0.7/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.7/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.7/fold0/gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.7/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/casey_confidence_cave/k10/conf0.9/fold0

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.9/fold0/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.9/fold0/gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/casey_confidence_cave/k10/conf0.9/fold0/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/cave_embedding_augmentation_conf0.7

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold0](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold1](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold1/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold2](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold2/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold3](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold3/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold4](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_sampled_fold4/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold0](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold1](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold1/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold2](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold2/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold3](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold3/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold4](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_flip_sampled_fold4/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold0](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold1](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold1/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold2](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold2/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold3](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold3/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold4](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_gray_sampled_fold4/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold0](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold1](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold1/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold2](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold2/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold3](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold3/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold4](presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_structured_low_sampled_fold4/README.md) | `graph_transformer` |

## presynaptic/cave_skeletons

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_fold0](presynaptic/cave_skeletons/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_fold0](presynaptic/cave_skeletons/gnn_lcpn_scratch_mean_resnet4x128_n10_fold0/README.md) | `mean` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold0/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold0](presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold0/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold0/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold1/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold1](presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold1/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold1/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold2/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold2](presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold2/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold2/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold3/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold3](presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold3/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold3/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold4/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold4](presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/fold4/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold4/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold0/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold0](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold0/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold0](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold0/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold0/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold1/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold1](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold1/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold1/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold1](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold1/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold1/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold2/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold2](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold2/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold2/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold2](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold2/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold2/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold3/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold3](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold3/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold3/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold3](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold3/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold3/README.md) | `pointwise_mlp` |

## presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold4/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold4](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold4/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_pre_post_fold4/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold4](presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/fold4/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_fold4/README.md) | `pointwise_mlp` |

## presynaptic/cave_v1dd_addition/no_v1dd

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/cave_v1dd_addition/no_v1dd/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/cave_v1dd_addition/no_v1dd/gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/cave_v1dd_addition/no_v1dd/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/cave_v1dd_addition/with_v1dd

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/cave_v1dd_addition/with_v1dd/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/cave_v1dd_addition/with_v1dd/gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0](presynaptic/cave_v1dd_addition/with_v1dd/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_mean_control

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_only

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_post_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_post_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold0](presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_mean_control

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_equal_sampled_pre_mean_control_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_only

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_only_matched_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_only_matched_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_equal_sampled_pre_only_matched_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_equal_sampled_pre_only_matched_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_equal_sampled_pre_only_matched_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_equal_sampled_pre_only_matched_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_post_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_equal_sampled_pre_post_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_equal_sampled_pre_post_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_equal_sampled_pre_post_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_equal_sampled_pre_post_fold0](presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_equal_sampled_pre_post_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_mean_control

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_mean_control_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n1_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_mean_resnet4x128_n1_mixed16_sampled_pre_mean_control_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_mean_control_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_only

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_only/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_only/gnn_lcpn_scratch_mean_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_only/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_only_matched_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_post_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n1_mixed16_sampled_pre_post_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n1_mixed16_sampled_pre_post_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_post/gnn_lcpn_scratch_mean_resnet4x128_n1_mixed16_sampled_pre_post_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold1/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold1](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold1/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold1/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold2/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold2](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold2/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold2/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold3/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold3](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold3/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold3/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold4/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold4](presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold4/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold4/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold0/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold0](presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold0/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold1/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold1](presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold1/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold1/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold2/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold2](presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold2/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold2/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold3/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold3](presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold3/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold3/README.md) | `pointwise_mlp` |

## presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold4/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold4](presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/fold4/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_fold4/README.md) | `pointwise_mlp` |

## presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_mean_control

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_mean_control/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_mean_control_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_only

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_only/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_only_matched_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_post_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_mean_resnet4x128_n17_mixed16_sampled_pre_post_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold0](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold0/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold0/README.md) | `pointwise_mlp` |

## presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold1/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold1](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold1/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold1/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold1](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold1/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold1/README.md) | `pointwise_mlp` |

## presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold2/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold2](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold2/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold2/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold2](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold2/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold2/README.md) | `pointwise_mlp` |

## presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold3/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold3](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold3/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold3/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold3](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold3/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold3/README.md) | `pointwise_mlp` |

## presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold4/pre_post

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold4](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold4/pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_fold4/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold4](presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold4/pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_fold4/README.md) | `pointwise_mlp` |

## presynaptic/new_skeletons

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_teasar_L4_H4_resnet4x128_n10_fold0](presynaptic/new_skeletons/gnn_lcpn_scratch_gt_teasar_L4_H4_resnet4x128_n10_fold0/README.md) | `graph_transformer_teasar` |

## presynaptic/new_skeletons_native/local_center/scale16/k17

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_fold0](presynaptic/new_skeletons_native/local_center/scale16/k17/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n17_fold0](presynaptic/new_skeletons_native/local_center/scale16/k17/gnn_lcpn_scratch_mean_resnet4x128_n17_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_fold0](presynaptic/new_skeletons_native/local_center/scale16/k17/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_fold0/README.md) | `pointwise_mlp` |

## presynaptic/new_skeletons_native/local_center/scale32/k9

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n9_fold0](presynaptic/new_skeletons_native/local_center/scale32/k9/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n9_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n9_fold0](presynaptic/new_skeletons_native/local_center/scale32/k9/gnn_lcpn_scratch_mean_resnet4x128_n9_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n9_fold0](presynaptic/new_skeletons_native/local_center/scale32/k9/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n9_fold0/README.md) | `pointwise_mlp` |

## presynaptic/new_skeletons_native/local_center/scale4/k65

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n65_fold0](presynaptic/new_skeletons_native/local_center/scale4/k65/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n65_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n65_fold0](presynaptic/new_skeletons_native/local_center/scale4/k65/gnn_lcpn_scratch_mean_resnet4x128_n65_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n65_fold0](presynaptic/new_skeletons_native/local_center/scale4/k65/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n65_fold0/README.md) | `pointwise_mlp` |

## presynaptic/new_skeletons_native/local_center/scale8/k33

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n33_fold0](presynaptic/new_skeletons_native/local_center/scale8/k33/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n33_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n33_fold0](presynaptic/new_skeletons_native/local_center/scale8/k33/gnn_lcpn_scratch_mean_resnet4x128_n33_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n33_fold0](presynaptic/new_skeletons_native/local_center/scale8/k33/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n33_fold0/README.md) | `pointwise_mlp` |

## presynaptic/new_skeletons_native/mixed_cell_b32/scale32/k9

| Run | Architecture |
|---|---|
| [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n9_mixed16_fold0](presynaptic/new_skeletons_native/mixed_cell_b32/scale32/k9/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n9_mixed16_fold0/README.md) | `graph_transformer` |
| [gnn_lcpn_scratch_mean_resnet4x128_n9_mixed16_fold0](presynaptic/new_skeletons_native/mixed_cell_b32/scale32/k9/gnn_lcpn_scratch_mean_resnet4x128_n9_mixed16_fold0/README.md) | `mean` |
| [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n9_mixed16_fold0](presynaptic/new_skeletons_native/mixed_cell_b32/scale32/k9/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n9_mixed16_fold0/README.md) | `pointwise_mlp` |
