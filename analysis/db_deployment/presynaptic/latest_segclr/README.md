# Newest SegCLR: six morphological classes, no confidence cutoff

Active scope follows the user's narrowed request: the six classes BasketFam,
BipFam, MartFam, NglFam, pyramidal and thalamocortical, with no 0.7 label-confidence
cutoff. Keep the original pointwise MLP (two layers) plus ResNet classification
trunk and the existing pre/post inputs. Train CAVE K=10 and native single-point
versions, five folds each (10 checkpoints). The three-class versions are excluded.

Pinned embedding model:
`resnet50_1280b_slab1280_50um_mac_partial_aug_2_style_bf16_100k__20261022_140504`,
checkpoint `checkpoint_e0_s100000`. The model requires two input channels and
uses the upstream CLAHE/segmentation volume registry preprocessing. Its source
checkout is pinned at `afb3d8c` in `logs/latest_segclr_src`.

The earlier production-wide geometry scan was cancelled. Focused planning uses
only training coordinates and local CAVE geometry whose content hashes match
the shared registry. Cached vectors must belong to the pinned checkpoint and
pass a GPU preprocessing comparison; remaining presynaptic/native and exact
postsynaptic points are generated through upstream SegCLR inference. Native
named-node inputs retain their original geometry and node IDs. The older compact
format, where present, retains its original mapping/reduction policy.

No old feature vectors or memory maps are substituted for newest-model inputs.
The cohorts and fold assignments match the original `conf0` manifests (2,029
cells). The first phase generates/assembles training features; the second trains
and registers the 10 classifier artifacts in `segclr_db`. This focused branch
does not run bulk production prediction generation.

Active job IDs are in `morpho6_jobs.json` and `jobs.json`. The superseded full
production submission is preserved in `jobs_superseded_full_deployment.json`.
A submission is not completion. The final registration receipt will be:

`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/latest100k_morpho6_centers_v1/classifier_registry.json`

The existing-SegCLR confidence-zero deployment and the original confidence-0.7
prediction import continue separately with their own IDs and progress records.

The active plan requests only the native centers consumed by the single-point
model, preserving the full geometry reference and original training windows.
Center-only feature files are explicitly marked and rejected for whole-window
use. The plan contains 15,904,686 unique input points: 11,169,025 cached and
4,735,661 missing newest-model embeddings. The prior 298.8-million-point plan
mistakenly included unused native geometry vertices and has been superseded.
The real-data representation check compares geometry, windows, training indices
and input tensors before the embedding swap; it gates GPU feature generation.
