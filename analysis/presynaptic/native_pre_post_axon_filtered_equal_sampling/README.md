# Filtered native pre/post with equal class sampling

Same filtered fold-0 cohort, model settings, and comparison figures as `../native_pre_post_axon_filtered/`. The nine submitted runs cover mean, pointwise MLP, and GT across pre only, raw pre mean control, and pre + post. Training uses `--class-balance equal`: per-window weights are 1/class_count, so each of the six L2 classes receives an expected 1/6 of draws. Sampling is with replacement and each epoch still draws the number of training windows. Test windows are evaluated without resampling. This changes sampling only; the loss remains unweighted.

Results are separate under `results/presynaptic/native_pre_post_axon_filtered_equal_sampling/scale16/k17/conf0.7/fold0/`. Open `pre_post_comparison.ipynb` to read progress and held-out metrics. All three conditions have been submitted.

Submit: `sbatch --array=0-8%2 scripts/sbatch/train_native_pre_post_axon_filtered_equal_sampling.sh`.
