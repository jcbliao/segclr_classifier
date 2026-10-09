# Incoming presynaptic fragments: Collina cohort, up to ten nodes

Output: `/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2`.
The original personal-scratch path is a symlink to this shared filesystem.

The frozen prior prediction source is the shared segclr_db `predictions` table,
version 14236, run
`resnet_hierarchy_v2__20260813_143332.checkpoint_244.cec46194ec1a.synapse_incoming_20000nm`.
Its `root_id` is the postsynaptic target; `subject_root_id` is the presynaptic
partner whose CAVE skeleton supplies the nodes. Only partner roots in prior
predictions onto neuronal targets are selected. All cached neuronal incoming
synapses from these selected roots are retained. The v2 cohort supplements the
previously missing ChC target with CAVE v1718 incoming synapses: 8,125,826
synapses and 3,282,796 partners. Selection follows partner root IDs rather than
the exact prior synapse subset. Provenance is frozen in `supplement_source.json`.

At root level, every selected partner has both a cached CAVE skeleton and
embeddings from `resnet_860b_reshuffled__20260603_150412`, checkpoint
`checkpoint_e0_s95000`. Exact selected-node coverage is checked during build.
No missing embeddings are generated or silently replaced.

Final build coverage: all 8,125,826 synapses were processed; all skeletons are
present. Selected-node embeddings are complete for 8,125,432 synapses.
The remaining 394 synapses across 341 presynaptic roots have at least one
missing selected-node vector (834 missing node instances across those
synapses). The latest embedding-table version is still the pinned version
488620, so a newer snapshot does not fill these gaps. These rows retain
`missing_embeddings` status, zero embeddings used, and no cell-type prediction.
Root details and synapse membership are recorded in
`missing_selected_node_embedding_roots.csv` and
`missing_selected_node_embeddings.parquet` under the output directory.

For each synapse, find the nearest node on its presynaptic partner skeleton,
then take up to ten nodes in shortest-path order along the skeleton, including
the center. Components with fewer nodes use all available connected nodes.
Mean the selected raw SegCLR vectors, and feed that mean to each of the five
geodesic 10-µm soma-exclusion classifiers. There is no inference soma mask,
matching-distance cutoff, proofreading filter, or minimum component-size filter.

`plan.json` pins all models and Lance table versions. Cohort shards and output
parts are Parquet. Each output row retains the synapse ID, target root ID,
partner root ID, anchor node ID and distance, selected original CAVE node IDs,
node count, embedding coverage status, 64D mean embedding, and five raw logit
vectors. Logit columns follow the explicit `logit_classes` schema metadata:
`astrocytic_process`, `axon`, `dendrite`, `soma`.

`n_embeddings_used` is the number actually averaged into the classified vector.
It ranges from 1 to 10 for classified fragments and is zero for unclassified
fragments. `n_embeddings_available` separately counts embeddings present at
the selected nodes; `node_count` includes selected nodes even if an embedding
is missing. `embedding_count_distribution.csv` and `summary.json` report the
distribution of actual contribution counts, for downstream evaluation by count.

The final `fragments.duckdb` provides `fragments` (all records) and
`classified_fragments` (complete inputs only) views over the Parquet parts.
`summary.json` reports all coverage failures. The shared segclr_db is read-only
for prediction facts throughout; the new derived database is separate. The
three Casey hierarchy metadata entries were registered in the shared database.

Jobs use account/QOS `mit_amf_advanced_cpu`, partition `mit_normal`, 32 array
shards with at most 16 running together, four CPUs each. The production array
depends on a real-data pilot, and finalization depends on every shard succeeding.
See `jobs.json` for current job IDs. Ten operational tests cover fragment
membership, missing vectors, model parity, hierarchical probability decoding,
Parquet round trips, per-fold filters, and distinct synapse preservation.

## Cell-type predictions

Every synapse retains five postsynaptic single-point subcompartment calls.
The presynaptic axon filter is hard and separate for each fold: the matching
subcompartment fold must classify the mean raw fragment embedding as axon.
Rejected records remain in the full table, with null cell-type predictions;
filtered views expose each of the twenty family/class-set/fold combinations.

Cell typing uses all five pointwise-MLP folds of CAVE pre/post n10 and single
pre/post, each with both three and six classes. CAVE n10 transforms each raw
node embedding with phi, then averages those transformed vectors; it does not
apply phi to the raw mean. Single pre/post uses the nearest center-node vector.
Both concatenate the exact corresponding postsynaptic point embedding.
Every model/fold also records its own `*_n_embeddings_used`: the actual
fragment count for CAVE n10, one for single pre/post, and zero when excluded.
The unprefixed count remains the raw fragment/subcompartment input count.

`cell_types/plan.json` pins twenty checkpoint hashes, manifests, hierarchy IDs,
class orders, raw local-head mappings, and aggregation selectors. Per-level
logits are conditional on each parent; explicit joint probabilities are saved
alongside them. Top-down class indices match the trained model decision rule.
Prediction identity for eventual segclr_db import is postsynaptic `root_id`,
presynaptic `subject_root_id`, presynaptic anchor `node_id`, and `synapse_id`.
Distinct synapses sharing a presynaptic fragment remain distinct predictions.

The parallel CPU join partitions postsynaptic inputs once before inference.
GPU inference batches fragments, vectorizes all five folds, prefetches reads,
and uses at most four simultaneous GPUs. Completed output parts are reused.
`cell_types/predictions.duckdb` and `cell_types/summary.json` are produced only
after all thirty-two shards finish and coverage/identity/filter counts agree.

Aggregation metadata is staged in a separate schema-v6 registry under
`cell_types/staged_registry`: `geodesic_mean_k10`,
`pointwise_mlp_mean_k10`, and `pointwise_mlp_mean_k1`. Each has `k_nearest`
and NULL `window_nm`. Learned pooling requires classifier weights and is
registered separately from raw means. Shared schema migration and prediction
import have not been performed.

## Completed run

All 32 fragment shards, five CAVE-six-class training folds, 32 cell-type
inference shards and final score validation completed successfully. The
independent completion audit also passed (job 25344851): every retained source
fragment field and postsynaptic field agrees across the full dataset; all
checkpoint hashes, output-part metadata, view paths and hierarchy hashes agree.

The dataset contains 8,125,826 distinct synapses onto 2,223 neurons, from
3,282,796 presynaptic partners. There are 160,190,532 included model/fold
prediction records across the four classifier families/class sets and five
folds. The 394 incomplete-embedding records remain explicitly excluded.

| Fold | Synapses passing the hard axon filter, for each classifier |
| --- | ---: |
| 0 | 8,006,943 |
| 1 | 8,009,607 |
| 2 | 8,014,751 |
| 3 | 8,001,919 |
| 4 | 8,014,413 |

Read `cell_types/predictions.duckdb` for the full table and twenty filtered
views. `cell_types/summary.json` gives model/fold counts;
`cell_types/completion_audit.json` records full-dataset verification;
`cell_types/real_data_parity_audit.json` records comparison against ordinary
model inference for all twenty cell-type models and all five pre/post
subcompartment folds on twelve real synapses. `writer_payload` in
`scripts/predict_incoming_fragment_cell_types.py` converts an included model/fold
batch to writer arguments with class names, logits, uncertainty and identities.
It does not write to the shared database.
