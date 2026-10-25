# Mapping saved predictions to segclr_db v7

Inspected 2026-10-16 against upstream main `40db1ec`, the local extensions,
the shared microns registry, and the frozen deployment plans. This is a
read-only assessment; no shared tables or database metadata were changed.

## What exists

The shared store at `/orcd/compute/sdorkenw/001/segclr-db/microns` reports
schema v7, materialization 1718. Its registry currently contains eight node
sets, seven prediction runs, and no classifier training-data links. None of
the seven prediction runs are our saved presynaptic cell-type or five-fold
subcompartment runs.

Our presynaptic deployment lives under
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2`.
Its completion audit records 8,125,826 synapses, 2,223 target cells, 3,282,796
partners, 1,664 prediction parts, and 160,190,532 included model/fold predictions.
There are 20 combinations: two model families, two class hierarchies, five folds.
The audit says these predictions have not been written to the shared database.
Import does not require repeating inference.

## Presynaptic cell-type row mapping

| Saved Parquet field | v7 destination | Meaning |
|---|---|---|
| `cell_root_id` | `predictions.root_id` | Postsynaptic target whose incoming synapses were queried |
| `partner_root_id` | `predictions.subject_root_id` | Presynaptic cell whose skeleton contains the anchor |
| `anchor_node_id` | `predictions.node_id` | Original CAVE skeleton node ID, without renumbering |
| `synapse_id` | `predictions.synapse_id` | Keeps separate synapses distinct even if they share an anchor |
| Constant `cave_skeleton_v4` | `predictions.node_set_id` | Identifies the geometry the anchor indexes |
| Per-level saved class index and frozen vocabulary | `pred_labels` | Preserve the saved top-down decision |
| One minus saved chosen-class joint probability | `pred_uncertainty` | Existing `writer_payload` convention |
| Per-level saved logits | `prediction_logits` | Preserve float32 values and class order |

The existing `writer_payload` in `scripts/predict_incoming_fragment_cell_types.py`
already performs this conversion except for the explicit `node_set_id` argument.
Filter each model/fold by its own `*_included` column. Missing/excluded rows
must remain exclusions rather than receiving placeholder predictions.

Keep one classifier run/checkpoint and prediction run per model/fold. Link
the actual SegCLR input run `resnet_860b_reshuffled__20260603_150412` and checkpoint
`checkpoint_e0_s95000`. Register exact checkpoint hashes, frozen model specs,
hierarchies and class order, fold splits and training-data provenance through
`link_run_data`. Do not fabricate training-data links when original label
versions or node-label provenance have not been established.

Use explicit prediction-run IDs that distinguish this deployment's input and
selection policy. The default derived ID does not include every input detail.
The model also consumes a postsynaptic embedding; a prediction row's anchor
identifies the presynaptic geometry and is not a complete description of the
two-sided input. Preserve the postsynaptic input source, fragment selection,
hashes, and fold-specific axon filtering in experiment specs/run notes and retain
the frozen Parquet/plan artifacts.

## Compatibility issues to address before import

1. **Local schema drift:** `store.schema_drift` reports
   `{'agg_specs': 'k_nearest, window_nm'}`. The local K-nearest extension adds a
   column and makes `window_nm` nullable; upstream v7's shared registry does not.
   The normal local `open_store` therefore rejects this store. Do not run our
   old v5-to-v6 helper: upstream's v6/v7 have different meanings.
2. **Learned pooling provenance:** the frozen plans name
   `pointwise_mlp_mean_k10` and `pointwise_mlp_mean_k1`, which are not registered
   in the shared store. These mean a learned MLP applied to each selected
   embedding followed by pooling, not a raw geodesic mean. A straightforward
   upstream-compatible representation is to put the entire learned pooling
   and K-selection configuration in the classifier experiment spec, leave
   `agg_spec_id=None`, and explain that choice in run notes. Alternatively,
   agree on a shared registry extension and migrate it explicitly. Do not
   relabel these as the existing 20-µm mean aggregation.
3. **Score semantics:** six-class saved logits scatter conditional local-head
   scores into global per-level vocabularies. A single softmax across a level
   does not recover the saved joint probabilities. V7 can store the logits,
   labels and uncertainty, but its prediction tables have no joint-probability
   column. Retain the Parquet probabilities and head topology, or extend the
   schema if direct probability queries are required. Import must not replace
   saved top-down labels with joint-probability argmax labels.
4. **Foreign-key validation:** verify target/synapse membership and each
   `(partner_root_id, cave_skeleton_v4, anchor_node_id)` against the current
   store. The writer's raw append primitive does not establish that every
   referenced node exists. Preserve source materialization and geometry IDs.
5. **Resume behavior:** import bounded groups by target cell and prediction run,
   using `add_cell_predictions` work tracking. An interrupted labels/logits
   write needs reconciliation before retry: a work marker alone does not make
   the two table appends atomic. Validate identities/counts on retry and avoid
   blindly appending existing parts.

## Five-fold subcompartment predictions

The separate frozen subcompartment plan describes 849,264,565 predictions
across CAVE, original TEASAR, and presynaptic-point geometries. This is planned
volume, not a fresh completion audit. Its cached four-class logits can also
be reused directly, with five independent classifier checkpoints and no
aggregation. Preserve its entropy-in-nats uncertainty convention separately
from the presynaptic cell-type convention.

The importer `scripts/subcompartment_predict_all.py` needs API changes:

- Replace `link_run_split` with `link_run_data` and verified training provenance.
- Replace `skeleton_name=` with `node_set_id=` in prediction writes.
- Replace named-point ingestion with registered node sets, `add_nodes` and
  the v7 embedding APIs. Preserve local node IDs and external synapse IDs
  separately (`point_ids` stores the external IDs).
- Replace reads of `named_skeletons` / `named_node_embeddings` with unified
  node-set-aware tables/APIs if rerunning preparation or inference.

The current node-set registry preserves TEASAR names verbatim, including
`teasar_existing_20260910`, `teasar_registered_20260911`, and
`teasar_testing_20260910`; dense `111nm` geometries have separate IDs. Match
each frozen item's original geometry exactly rather than substituting its
dense counterpart. The inspected registry has no presynaptic-point node set;
those points need a deliberate node-kind definition because the currently
accepted non-skeleton kind is named `postsynaptic_points`.

## Suggested implementation order

Resolve upstream/local registry compatibility first. Then build a read-only
preflight producing exact model/run IDs, node-set mappings, registry additions,
foreign-key coverage and expected per-run label/logit counts. Exercise the
conversion in a temporary v7 store and compare saved labels, uncertainty,
logits and synapse identities. Finally stream the existing artifacts into the
shared store with bounded writes and resumable reconciliation. Keep the
original artifacts as the full record of probabilities and fragment inputs.
