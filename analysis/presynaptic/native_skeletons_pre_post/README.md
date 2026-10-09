# Native skeleton pre/post analysis

Open `pre_post_comparison.ipynb` for the native bin16/k17, confidence >=0.7
comparison of mean, pointwise MLP, and GT with and without a postsynaptic vector.
The notebook follows the CAVE embedding augmentation comparison template.

It reads epoch CSV files, best-so-far metrics, and final result JSON files from
`results/presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/fold*`.
Final metrics take precedence; incomplete runs are explicitly labeled. Rerun
all cells to refresh training progress. No checkpoints or GPU are required.

Figures cover training curves, paired held-out metrics and differences,
cell/window per-class recall, and normalized confusion matrices with raw counts.
The excluded-site table preserves root IDs as strings and reports coordinates
in nanometres. Presynaptic window uniqueness is unchanged by post-point identity.

Matched-width controls append the raw 64-dimensional presynaptic mean after
pooling. The notebook compares all three conditions and reports post-minus-pre
and post-minus-control differences, prioritizing window macro F1.
