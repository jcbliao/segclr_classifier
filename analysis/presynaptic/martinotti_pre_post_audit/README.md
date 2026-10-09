# Martinotti test-cell window vote audit

Compare completed native pre/post runs in the original, dendrite-filtered, and equal-sampling experiments. Only true `MartFam` held-out cells are re-inferred from best checkpoints. Each cell prediction is the plurality of its window labels, matching the training evaluation (ties choose the first class index).

`martinotti_called_pyramidal.csv` lists each true Martinotti cell called pyramidal per run, with counts and fractions for Martinotti, pyramidal, and all other classes. `all_martinotti_cells.csv` includes all 23 true Martinotti cells per run. `population.csv` reports window-weighted fractions across true Martinotti test cells and all test windows, plus equal-cell averages where inference is complete.

CPU re-inference uses float32; saved training evaluation used CUDA bfloat16 AMP. The audit checks matching window totals, class count differences below 0.2 percentage points, and exact agreement with the saved Martinotti cell-level confusion row. The recorded per-run difference quantifies the precision effect. Fractions represent hard class vote shares, not predicted probabilities.

Run `segclr_db/.venv/bin/python analysis/presynaptic/martinotti_pre_post_audit/analyze.py` to refresh. Completed run CSVs are cached. Use `--summarize` to rebuild tables without inference. Incomplete training runs are omitted.
