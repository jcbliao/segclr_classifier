# Presynaptic-seeded 20 µm experiment

Completed October 26, 2026: all five folds trained for 30 epochs and exported
all five local heads for 1,744,210 held-out unique windows in total. The final
certification verifies epoch coverage, finite checkpoint weights, configuration
hashes, head identities and prediction row counts. See
[`completion.json`](../analysis/presynaptic/latest100k_geodesic20um/completion.json)
for checkpoint paths, per-fold metrics and local-head metrics conditioned on
the true parent. Job IDs are recorded alongside it in `jobs.json`.

This experiment changes the latest 100k SegCLR CAVE model from ten nearest
embeddings to **all observed embeddings within 20,000 nm geodesic distance**.
Every seed is an existing presynaptic site, snapped to its existing nearest
observed CAVE skeleton node. The radius is measured from that anchor, inclusive
of the boundary. Spatially adjacent branches do not join unless the skeleton
path is within the radius. Unobserved nodes can carry paths but are not averaged
as zero embeddings. There is no maximum embedding count or ten-node minimum.

The source's per-site anchor eligibility and soma geometry are retained: the
5 µm cut applies where present; source records explicitly marked
`soma_cut_applied=False` retain their full soma-free skeleton. The database export's
`canonical_skeleton_audit.json` records this flag per cell: 2,016 have the cut and
13 retain the full skeleton.
An audit found that the source metadata declares a 2 µm match cutoff, while
some cell records accept anchors up to 5 µm away. We preserve the actual
`cave_within_match_cutoff` decisions rather than silently discard those sites.
This is an anchor-centered skeleton distance, not a 20 µm Euclidean sphere or
20 µm total cable-length budget. The off-skeleton snap distance is recorded in
the original arrays and is not added to the geodesic radius.

The model retains the latest pointwise MLP (two layers), ResNet classifier
trunk, postsynaptic embedding, confidence threshold zero, five cell splits,
balanced sampling, and 30 epochs. At the user's request, it restores the two
finest morphology hierarchy levels (`hierarchy_levels_dropped: 0`) instead of
truncating at six classes. All branching LCPN heads are trained, with separate
conditional logits for neuron, excitatory, inhibitory, pyramidal, and IT parents.
Single-child branches are deterministic passthroughs. Inputs use SegCLR
`resnet50_1280b_slab1280_50um_mac_partial_aug_2_style_bf16_100k__20261022_140504`,
checkpoint `checkpoint_e0_s100000`. Preparation reuses the latest raw cached
vectors; it does not generate embeddings or modify the shared database.

The existing per-cell membership-set deduplication and postsynaptic eligibility
still apply. Changing window membership changes which sites deduplicate. Short
components previously rejected for having fewer than ten embeddings become
eligible if their existing postsynaptic feature is available. The generated
plan reports counts before and after these filters. Thus the cell split is
matched, while window counts need not equal K10 counts.

## Run artifacts and launch

- Preparation and GPU preflight: `analysis/presynaptic/latest100k_geodesic20um/`.
- Radius data: `/orcd/scratch/orcd/013/jcbliao/presynaptic_axons_latest100k/geodesic20um_sourceanchors/conf0/`.
- Training: `results/presynaptic_latest100k/geodesic20000nm/full_hierarchy/conf0/fold{0..4}/`.
- Every-head classification: `analysis/presynaptic/latest100k_geodesic20um/all_lcpn_predictions/fold{0..4}.parquet`.
- Final certification: `analysis/presynaptic/latest100k_geodesic20um/completion.json`.
- Source configurations: `analysis/db_deployment/presynaptic/latest_segclr/morpho6/configs/00.json` through `04.json`.

```bash
prepare=$(sbatch --parsable scripts/sbatch/prepare_presynaptic_radius.sh)
smoke=$(sbatch --parsable --dependency="afterok:$prepare" --time=00:30:00 \
  --job-name=pre_geo20_smoke scripts/sbatch/train_presynaptic_radius.sh smoke)
sbatch --array=0-4%2 --dependency="afterok:$smoke" scripts/sbatch/train_presynaptic_radius.sh
```

Preparation tests radius boundaries, disconnected branches, unobserved bridges,
duplicate edges, short components, and shortest-path agreement against SciPy.
The GPU preflight exercises the real training loader with the smallest and
largest bags, compares model pooling to an explicit mean of transformed nodes,
and checks finite forward/backward computations. It also verifies every local
head against its direct forward pass and round-trips the score columns through
Parquet. Existing K10 aggregator/trunk weights and fresh full-hierarchy heads
are used only for this preflight; the radius experiment trains fresh weights.

Each fold automatically exports classification after training. Every held-out
unique window gets every local head's logits, conditional probabilities and
local class index, plus all levels' joint probabilities and top-down predictions.
The sidecar names each head's parent and ordered children. Scores are retained
even for heads outside the top-down path. These are held-out-fold outputs from
the same fold used for checkpoint selection, not an additional untouched test
set. Original source synapse IDs identify the deduplicated window representatives.

The compatibility field `cave_valid_k_window` means a valid radius bag in this
new format. `metadata.json` explicitly identifies radius selection. Training
requires `num_embeddings: null` and the matching radius; fixed-K configs fail.
Geometry features are not prepared for this set-only experiment, so geometry
models and stale memory maps are rejected.

## Review of Collin's pipeline

Reviewed upstream `dorkenwald-lab/segclr_db` main at **eb6e297** (2026-10-25),
including `docs/PIPELINE.md`, `src/pipeline*.py`, and
`docs/INFERENCE_OPTIMIZATION_20261025.md`.
The installed local checkout is left intact because it includes local schema
and hierarchy work.

Useful upstream changes include spatial batches of approximately 100 target
cells, one partner-skeleton load per batch, CPU planning with independent GPU
consumers, resumable file-locked embedding batches, and complete-window
coverage checks. The October 25 storage changes reduce large membership-scan
costs and index rewriting. Those paths are appropriate for generating additional
raw embeddings when deploying the radius model.

The stock prediction flow is **not directly compatible** with our model:

| Upstream path | Required adaptation |
| --- | --- |
| `pipeline_config.py` resolves the `cell_classification` training schema and requires geodesic raw-mean aggregation | Recognize this repository's checkpoint/config, pin embedding provenance, radius and skeleton-selection policy |
| `pipeline.py::_aggregate_batch` stores raw mean vectors at partner anchors | Pool `phi(raw_embedding)` per classifier checkpoint, or pass raw ragged bags to the predictor |
| `pipeline_models.Classifier.classify(vectors)` accepts one pooled vector | Supply ragged raw neighborhoods plus each synapse's postsynaptic vector |
| Prediction stages include target skeleton nodes and incoming synapses | Add a synapse-only mode for this experiment; do not create arbitrary skeleton-node training windows |
| Partner windows expand canonical full skeleton anchors | Match this experiment's observed-node snapping, soma cut and raw-embedding coverage policy explicitly |

Our required computation is `head(concat(mean(phi(pre_embeddings)), post_embedding))`.
Applying `phi` to an already averaged raw vector changes the function. The
postsynaptic input also prevents reusing the final answer solely by presynaptic
anchor: two synapses sharing an anchor can have different postsynaptic vectors.
Transformed presynaptic nodes can be reused within each fold/checkpoint, but
must not be registered as ordinary SegCLR embeddings or raw geodesic means.
Preserve native conditional head logits and hierarchical probabilities when
writing predictions; this head has no SNGP covariance estimate.

The radius model uses the existing trainer. Exporting its frozen held-out
predictions is separate from integrating its predictor into upstream's production
pipeline. The upstream batching/embedding flow can be reused after implementing
the interfaces above and checking prediction parity.

## Exporting frozen results to segclr-db

The export uses an isolated upstream SegCLR DB checkout, leaving the local
editable installation intact. The validated revision is `eb6e297`:

```bash
git clone https://github.com/dorkenwald-lab/segclr_db logs/segclr_radius_export_upstream
git -C logs/segclr_radius_export_upstream checkout --detach eb6e297
```

Alternatively, set `SEGCLR_EXPORT_CHECKOUT` to an existing compatible checkout.
Training configurations and provenance snapshots record their original lab
filesystem paths. Execution requires the referenced lab data and checkpoints;
the Git repository retains code, notebooks, configurations, and receipts, while
weights and per-window prediction Parquet files remain in the shared stores.

Export completed October 26, 2026: all five models are registered, with
**1,744,210 predictions and 8,721,050 level-logit rows** verified against the
frozen outputs. The local receipt and its shared copy contain the five run IDs.

`scripts/sbatch/export_presynaptic_radius.sh` publishes the five saved folds into
the shared `microns` dataset at `/orcd/compute/sdorkenw/001/segclr-db`. It copies
weights, model configurations, manifests, and the original all-head probability
and logit files into a shared artifact bundle before registering the models.
It does not run inference on additional synapses. The export population remains
the 1,744,210 held-out, membership-deduplicated window representatives.

Future exports automatically use as many fold workers as the allocation supports,
up to the number of folds. Each worker reserves four CPUs and 8 GiB, plus 8 GiB
for coordination. Scanner read-ahead and I/O buffers are bounded: a full-fold
read-only benchmark peaked at 2.7 GiB, and the five-worker fixed-ten production
export peaked at approximately 20 GiB. The standard batch job requests
20 CPUs and 48 GiB so all five folds can publish and verify concurrently with
memory headroom and without requiring a large-memory node.
`--workers N` optionally caps this; smaller allocations automatically reduce the
worker count. Each worker opens its own database handle in a fresh process.
Registration runs once before the workers start, and only the parent writes
progress/completion receipts. Retries repair only the affected prediction run;
the completion receipt requires every fold to pass exact verification.

The same workflow also supports `--state` for the 10 µm and fixed-ten experiments:

```bash
sbatch scripts/sbatch/export_presynaptic_radius.sh \
  --state analysis/presynaptic/latest100k_geodesic10um
sbatch scripts/sbatch/export_presynaptic_radius.sh \
  --state analysis/presynaptic/latest100k_k10_full_hierarchy
```

Each experiment has separate configurations, checkpoints, prediction run IDs,
receipts, and a `five_fold_results.ipynb`. The fixed-ten variant reuses the
previous nearest-ten observed embedding windows, while restoring all morphology
levels; it is not a ten-edge hop radius. Its legacy `window_nm` config field does
not define membership. The registry records `k_nearest=10` without a radius.

Both additional runs completed October 26, 2026: five folds and 30 epochs per
fold, with all local-head scores exported and verified in segclr-db using five
concurrent workers. The [10 µm notebook](../analysis/presynaptic/latest100k_geodesic10um/five_fold_results.ipynb)
covers 1,746,215 held-out windows; the [fixed-ten notebook](../analysis/presynaptic/latest100k_k10_full_hierarchy/five_fold_results.ipynb)
covers 1,608,393. Both notebooks were executed successfully. Their adjacent
`segclr_db_export/receipt.json` files contain the model and prediction run IDs.

Before writing predictions, the exporter verifies source hashes, canonical CAVE
skeleton content and soma-cut index mappings, biological synapse identities,
class ordering, and conditional probabilities. An isolated database pilot checks
all five models, recovery after an interrupted label/logit write, repeat imports,
and the public read API. Every published fold is compared exactly with the saved
labels, uncertainty, and logits. A successful export writes
`analysis/presynaptic/latest100k_geodesic20um/segclr_db_export/receipt.json`;
`complete: true` certifies all five folds were verified.

Database `root_id` identifies the postsynaptic target, `subject_root_id` the
classified presynaptic cell, and `node_id` the canonical presynaptic CAVE anchor.
`synapse_id` distinguishes multiple sites sharing an anchor. The source Parquet's
`root_id` is the presynaptic cell; the shared `anchor_mappings` files preserve
the explicit translation.

The database stores one raw logit vector per hierarchy level. Each local head is
scattered into that level's registered class order; deterministic unary entries
are zero. **Apply softmax separately to each head's `child_global_indices`, not
to an entire hierarchy level.** Every head is stored even when its parent is
outside the predicted path. The shared source Parquet also preserves each head's
conditional probabilities and each level's joint probabilities. Stored
`pred_uncertainty` is `1 - joint_probability[selected_class]`, not SNGP variance.

```python
import json
from pathlib import Path
from segclr_db import SegCLRDatabase

receipt = json.loads(Path(
    "analysis/presynaptic/latest100k_geodesic20um/segclr_db_export/receipt.json"
).read_text())
assert receipt["complete"]
db = SegCLRDatabase(root=receipt["database_root"], dataset="microns")
model = receipt["models"][0]  # fold 0; one separate prediction run per fold
result = db.get_predictions(
    model["prediction_run_id"], root_ids=[postsynaptic_root_id],
    with_logits=True, node_set_id="cave_skeleton_v4",
)
head = model["local_heads"][0]
conditional_logits = result.logits[head["child_level"]][:, head["child_global_indices"]]
```

In this shared store, `return_coords=True` currently encounters an existing
restricted scalar index on `nodes`. The exporter checks coordinates against the
same table using `use_scalar_index=False`, without changing shared permissions.
Labels and logits can be retrieved with the call above.
