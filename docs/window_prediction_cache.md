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
`analysis/all_windows/feature_prediction_correlation.py` uses this cache. Either batch task
can create it. The Neuroglancer task reads only rows whose split is `test`;
training-fold predictions are never generated for or included in that layer.

## Resumable Neuroglancer export

Submit the complete inference, parallel conversion, and publication chain with:

```bash
scripts/submit_neuroglancer_predictions.sh [RUN_NAME ...]
```

The default is 16 CPU array shards; override it with `SHARDS=32`. The prepare
job reuses every valid test prediction cache and freezes an immutable plan
under `segclr_predictions/fold_test_validation/builds/<plan-id>/`. Array tasks
own disjoint root IDs from the held-out whole-cell test fold
and publish each binary with a temporary-file rename, so preemption cannot leave
a partial payload. Resubmitting a shard skips payloads it already completed.

The dependent finalize job verifies that every planned root exists and is
non-empty. It then installs metadata and swaps the completed skeleton directory
into place. An incomplete build is never published as the live Neuroglancer
source. The previously published source is retained as `skeletons.previous`.
The folder name, export plan, prediction-label manifest, and per-segment `fold`
property all identify this as the `test/validation` fold. In this project,
validation is an alias for test rather than a separate third partition.

Discrete predictions are stored as four-bit values (`unavailable=-1` becomes
zero; class codes `0..7` become `1..8`). Six predictions fit exactly in the 24
integer bits of one float32, and four floats form each `vec4` attribute, for 24
models per WebGL attribute. With radius and `target_class` occupying two of the
16 available attributes, one skeleton source can therefore carry 336 models without
duplicating geometry. `prediction_labels.json` records each model's attribute,
component, nibble/divisor, and the GLSL decoding expression.

Generate a paste-ready Neuroglancer layer JSON and editable skeleton shader for
one model with:

```bash
MODEL=mpnn_L2_position_lpe_resnet4x128_n20 MODE=rainbow \
  sbatch scripts/sbatch/make_neuroglancer_prediction_shader.sh
MODEL=mpnn_L2_position_lpe_resnet4x128_n20 MODE=correctness SEGMENT=ROOT_ID \
  sbatch scripts/sbatch/make_neuroglancer_prediction_shader.sh
```

The generated shader begins with plainly named `COLOR_<TYPE>` and
`OPACITY_<TYPE>` constants. Skeleton edges expose RGB but not per-vertex alpha,
so fractional values use screen-door transparency and zero hides a type; the
layer-level selected opacity remains one.

Generate a compact, percent-encoded Spelunker link selecting every held-out
neuron of one ground-truth type with:

```bash
MODEL=mpnn_L2_position_lpe_resnet4x128_n20 TYPE=pyramidal MODE=rainbow \
  sbatch scripts/sbatch/make_neuroglancer_prediction_link.sh
```

The link uses 8 nm x 8 nm x 40 nm dimensions and the supplied CAVE-viewer
camera template. The ordinary CAVE segmentation layer selects no objects; all
matching roots are selected only in the prediction skeleton layer, with the
generated shader already installed. `MODE=correctness` is also supported, and
`MODE=both` adds separate rainbow and correctness skeleton layers containing
the same roots so they can be toggled and edited independently.
