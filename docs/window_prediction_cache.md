# Shared window-prediction cache

All baseline per-window inference is stored outside any particular analysis or
visualization product at:

`/orcd/scratch/orcd/013/jcbliao/segclr/window_prediction_cache`

There is one compressed NPZ named `<run_name>.npz`. Each row identifies its
split, root ID, graph-cache center index and center position, and contains the
finest-level prediction and target. `schema.json` in that directory documents
the arrays for consumers outside this repository.

Within this repository, use `data.window_prediction_cache.load_prediction_cache`
instead of opening files directly. Set `SEGCLR_PREDICTION_CACHE` to override the
shared location for testing or another deployment.

Both `scripts/export_neuroglancer_predictions.py` and
`analysis/feature_prediction_correlation.py` use this cache. Either batch task
can create it; the Neuroglancer task expands a test-only cache to train+test.
