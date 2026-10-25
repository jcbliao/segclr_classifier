# CAVE presynaptic segmentation augmentation

Preparation and CPU correctness tests are complete. The original smoke job
26115061 was cancelled while pending; expanded benchmark job 26115419 is submitted. `pilot_plan.json` freezes the existing conf0.7 fold0
CAVE presynaptic training cohort selection and source geometry/checkpoint hashes.
It selects 128 nodes in each of three cells: 384 crops, 2,688 embedding rows.
This is a smoke/throughput pilot, not a representative full-cohort distribution.
The source manifest/database are taken from the saved six-class CAVE pointwise
MLP pre/post classifier's fold0 result record. The full fold has 1,414 training
cells and 359 held-out cells. Only presynaptic CAVE embeddings are augmented.

## Proposed experiment

Use the existing parallel crop reader with pinned double buffers and Morton
ordering, grouped model forwards, and two bounded background Parquet writers.
Read raw 129-cubed EM once and a 135-cubed mask once for each crop. Generate a
clean baseline and 1/2/3 layers of erosion and dilation, updating all voxels
simultaneously using six face-sharing neighbors. Reuse preceding layers;
optionally compile the Boolean neighborhood kernel to fuse GPU operations.
The halo removes artificial crop-edge effects. Out-of-volume voxels remain
background, and final masks exclude segmentation padding. No masks are saved.
Different Z spacing is recorded; these are voxel layers, not isotropic nm radii.

The original 129-cubed crop stays fixed. Expansion exposes additional raw EM
within that crop, and shrinking removes foreground. A completely eroded mask
is still embedded and explicitly counted as empty; it is not silently dropped.

## Outputs

- `parts/<root_id>.parquet`: 64-D float32 embeddings, original CAVE node IDs,
  augmentation IDs, baseline/augmented counts, count differences, halo flags.
- `embeddings.duckdb`: persistent view over the Parquet files. Keep the parts
  with this file; the DuckDB file is not a self-contained copy.
- `mask_statistics.csv`, `summary.json`, histogram Parquet and per-cell CSV:
  voxel counts, signed/fractional changes, quantiles, occupancy/physical volume,
  empty/clipped counts, and baseline embedding cosine similarity.
- `timings/<root_id>.json`: CUDA transfer, morphology, mask/count, model and
  download timings, plus CPU output-writing time. Compilation/startup is
  included in generation timings; the synthetic benchmark uses warmups.
- `training_choices/segmask/cells/<root_id>.npz`: existing presynaptic-loader
  format with clean plus all augmented views. `train_gnn.py` accepts `segmask`.

Pilot exports cover only the sampled nodes and cannot train the complete
classifier. Production preparation must use `--nodes-per-cell 0 --limit-cells 0`
with a new plan and output directory. Fold0 held-out cells stay clean. The
manifest can be overridden for another fold; do not silently mix fold cohorts.
This pipeline augments the presynaptic CAVE embeddings; postsynaptic inputs are
unchanged. It does not write to the shared segclr_db.

## Before production

Review the proposed neighborhood and strengths, then run the bounded GPU pilot:

```bash
sbatch scripts/sbatch/segmentation_augmentation_pilot.sh
```

The launcher first benchmarks eager and compiled morphology against inference
and compares their masks exactly, then generates and summarizes real pilot
crops. GPU timing and throughput have not yet been measured. Start with four
reader threads and 16 crops/batch, grouping four variants per forward. Use the
pilot to select batch size, reader count, compilation and grouping before
preparing a balanced multi-GPU production plan via `--tasks`.

The production launcher is intentionally not submitted automatically. Full
generation needs a reviewed cohort/output plan and the pilot's timing and
numerical checks. Import into standard v7 augmentation tables, if desired,
should follow output validation and the explicit aggregation-registry update.

## Expanded benchmark

`benchmark_plan.json` selects 512 nodes from each of 32 cells, balanced across
cell types (deterministic seed). It generates 114,688 embeddings from 16,384
crops. `benchmark_clean_plan.json` uses exactly the same nodes for a separate
clean-only pass. Both use batch 32, FP16 and compiled inference; the augmented
pass groups four views and compiles morphology. Job 26115419 runs both passes,
summarizes each, and exports augmented training choices. This sample tests
sustained throughput; its balanced cell-type distribution is not a population
frequency estimate.

## Full production submission

Submitted preparation job 26117443 for every eligible node in the 1,414
fold0 training cells (no per-cell node cap). GPU arrays 26117457 (shards
0–11, up to four preemptable GPUs) and 26117458 (shards 12–17, up to two
normal GPUs under the available AMF advanced account) depend on preparation.
Finalization job 26117459 depends on both complete arrays and generates
statistics plus training-choice exports. Outputs are under
`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/segmentation_augmentation/cave_conf0.7_fold0_n6_r3_20261016`.
See `full_jobs.json` for job IDs/settings; exact node counts will be recorded
in `full_plan.json` when preparation completes. Held-out cells stay clean.

## Expanded production scope: held-out cells included

The training-only jobs above were cancelled before starting. Replacement
preparation 26117501 includes all 1,773 manifest cells (1,414 training plus
359 held-out). Arrays 26117502 and 26117503 run all nodes in 18 shards, up to
six GPUs concurrently. Finalization 26117504 builds statistics and training
exports. Split assignments are retained in each plan item. The plan is
`full_all_cells_plan.json`; outputs use the distinct shared directory
`cave_conf0.7_fold0_all_cells_n6_r3_20261016`. Generating held-out embeddings
does not change the classifier split assignments or automatically train on
held-out examples. Current job metadata is in `full_jobs.json`.

## Recovery and normal-GPU-only scheduling

Shards 12/13 completed (673,533 nodes). Shards 14–17 failed during CUDA
initialization on node3006 and are retried by array 26130404 with that node
excluded. Array 26130456 replaces the pending preemptable shards 0–11.
Every active production GPU job now uses mit_normal_gpu. The available advanced
GPU QOS limits aggregate concurrency to four GPUs. Finalization is 26130457;
current IDs and coverage remain in full_jobs.json. The full plan contains
6,061,803 nodes and 42,432,621 embeddings.
