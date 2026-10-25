# Presynaptic deployment without the 0.7 label-confidence cutoff

This version trains `cave_n10` and `single_pre_post`, each with three-class and
six-class heads and five folds. The existing `conf0` cohort contains 2,029 cells
versus 1,773 at `conf0.7`. Original fold assignments and labels are preserved
for all shared cells. Architecture, training settings, postsynaptic inputs,
class grouping and the deployment's fold-specific hard axon filter are retained.
The existing four confidence-filtered configurations remain separate.

Preparation verifies the cohorts, aligns postsynaptic embeddings for all cells,
reuses matching verified caches, and freezes 20 training configurations with
source template and manifest hashes. It checks real dataset loading and feature
alignment before dependent training can start. `plan.json` records all model
paths, configs and hashes. Each training task requires a CUDA GPU.

The dependent pipeline trains 20 checkpoints (four concurrent GPU tasks), freezes
the inference plan, checks all 20 models' vectorized inference against their
training forward/head path on GPU, predicts 32 source shards, and audits the
saved rows. Inference reuses the frozen original fragments and matched
postsynaptic embeddings. It does not relabel or modify the old deployment.

After the prediction audit succeeds, the importer verifies exact synapse/node
foreign keys and temporary-store round trips, then registers shared model
artifacts and imports the new labels/logits into `segclr_db` with separate IDs.
The new deployment root is:

`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_conf0_v1`

`jobs.json` records the submitted dependency chain. A submitted or trained run
is not a completed database import. Import progress will be at
`logs/presynaptic_import_conf0/progress.json`; the final shared receipt will be
`cell_types/segclr_db_import.json` under the new deployment root.

Submit a fresh pipeline with:

```bash
bash scripts/submit_conf0_presynaptic_deployment.sh
```

An optional first argument reuses an already-submitted preparation job.
The launcher refuses duplicate submission when `jobs.json` already exists.

Preparation completed successfully for all 20 configs and all 2,029 cells.
Config replay omits optional null flags and exercises the actual training CLI
parser during preparation. GPU jobs exclude `node3006`, whose allocated CUDA
devices failed initialization during the first launch. Superseded submission
records are preserved as `jobs_attempt*.json`; `jobs.json` is the current chain.
