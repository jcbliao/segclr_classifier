# Five-fold whole-cell window-size comparison

Completes missing folds for mean pooling, two-layer pointwise MLP, two-layer MPNN without geometry inputs, and two-layer/four-head GT with its original geometry features, at 10, 20, and 40 nodes. Every model uses the original four-block width-128 ResNet classification head and original CAVE labels. Source windows cover the whole cell with the existing soma exclusion.

The earlier report sweep covered the 10-node head comparison, leaving larger windows at fold 0. This extension reuses that sweep's cell-held-out fold manifests, existing completed reports, and each size/model's historical training settings (including batch size, epochs, sampling, optimizer and dropout). Only missing reports are trained: 40 runs. No original reports are overwritten.

`plan.json` lists all 60 comparison reports, the 40 missing jobs, their source templates, and 12 shared packing groups. Configurations are under `configs/`. `build.py` validates configurations and source membership caches before freezing the plan, and refuses to rebuild a submitted plan. `run.py` handles CPU packing or one GPU training task.

Fast settings: AMP, optimized loading, shared CPU packing, GPU residency where memory permits, and CUDA graphs for supported models. MPNN/dropout and oversized GPU inputs use the trainer's compatible paths. CPU packing permits up to 256 GiB estimated packed size for the larger whole-cell windows. GPU jobs require completed CPU caches rather than rebuilding on GPU nodes. Six-hour training segments requeue with checkpoint resume at the time-limit warning.

Submission IDs and dependencies are recorded in `jobs.json`. Packing uses up to four 16-CPU jobs; training uses up to four L40S GPUs. Per-task stdout/stderr are in `logs/window_size_folds_<job>_<task>.*` and `logs/pack_window_sizes_<job>_<task>.*`. Final metrics and checkpoints remain under `results/all_windows/` with fold-specific names. Prediction caches are isolated under `/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/window_size_five_folds/predictions/`.

These folds retain the historical checkpoint selection on the held-out fold; they are not a separate untouched final test set. Five-fold summaries should wait until every corresponding report exists.
