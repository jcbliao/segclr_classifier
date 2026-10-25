# Five-fold CAVE versus TEASAR confidence comparison

150 combinations: two skeleton pipelines × mean / pointwise MLP / full GT × confidence thresholds 0, 0.3, 0.5, 0.7, 0.9 × five folds. Thirty completed fold-0 reports are reused; 120 missing runs are launched. CAVE uses 10 observed nodes; resampled TEASAR uses scale16 / 17 nodes. This compares the established pipelines, not a skeleton-only ablation with equal node counts.

Both pipelines use the same six Morpho neuron classes (BasketFam, BipFam, MartFam, NglFam, pyramidal, thalamocortical), cells and root-level train/test assignments at each threshold/fold. Folds are stratified using Morpho fine labels, assigned once across nested confidence cohorts. A root keeps its held-out fold wherever it appears. Thalamocortical cells retain their original labels and are included at every threshold because Morpho coarse confidence is unavailable for them. These are the existing presynaptic datasets with presynaptic embeddings only; no postsynaptic embeddings or new subcompartment filtering are introduced in this sweep.

Every model trains end to end from scratch (or resumes its own matching incomplete checkpoint), with the existing 4×128 ResNet classification trunk, class-balanced sampling, 30 epochs, seed 0, batch size 4096, mixed-cell batch size 16, attention budget 14,450,688, Adam learning rate 1e-4 and weight decay 1e-5. GT retains LPE, relative position and adjacency bias. Mean and pointwise MLP use embeddings only. Nothing is frozen. Resources are 16 CPUs / 15 loader workers / 64 GB / one L40S per run, maximum four normal plus four preemptable training GPUs across both skeleton pipelines. The known CUDA-failing node3006 is excluded.

The historical trainer selects checkpoints on its held-out fold and reports that same fold. Thus these are cross-fold model-selection scores, not estimates on an additional untouched test cohort. Comparing confidence thresholds also changes the evaluated population. Do not attribute a threshold effect solely to label quality without considering cohort composition.

The pre-submission audit checks all 50 manifests and metadata, matching CAVE/TEASAR roots/labels/splits, hierarchy classes, nested cohorts, stable root assignments, exactly-once held-out coverage across five folds, all source cell NPZs and all TEASAR memmap completion markers. All 120 configurations pass the current training CLI parser. Original fold-0 argument records are the templates; changes are restricted to fold-specific paths, frozen manifests and CPU worker count. Existing reports are checked before reuse. New runs write into the established `confX/foldY` result folders without overwriting completed fold-0 reports.

Files:

- `plan.json`: all 120 new jobs, 30 reused reports, source/config/manifest hashes and cohort audit.
- `configs/`: frozen per-job argument JSONs.
- `manifests/`: frozen labels, hierarchy and root splits for both skeleton pipelines.
- `jobs.json`: training wave IDs, continuation IDs and final report audit.

Launchers: `scripts/launch_cave_teasar_confidence_sweep.py` builds/audits the plan; `scripts/run_cave_teasar_confidence_sweep.py` verifies hashes and executes one job. `scripts/sbatch/train_cave_teasar_confidence_sweep.sh` runs array tasks. Do not rebuild the plan while jobs are using it.

The GPU partition has a 64-submitted-job per-user limit. Four automatic waves of 30 jobs keep submissions below that limit and at most four GPUs active. After a wave terminates, its dependent CPU continuation submits the next wave and schedules another continuation. The final continuation generates both existing five-fold summary CSV sets and records missing reports explicitly if any run failed. Progress and every submitted wave are retained in `jobs.json`. CPU continuation jobs use `mit_amf_advanced_cpu`.

CAVE result root: `results/presynaptic/morpho_confidence_cave/k10/`.
TEASAR result root: `results/presynaptic/morpho_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/`.

## Partition allocation update

Exactly 30 of the 120 new tasks are assigned to `mit_preemptable` under `mit_general` / `normal` QoS; 90 use `mit_normal_gpu` under `mit_amf_advanced_gpu`. Concurrency is three normal GPUs plus one preemptable GPU. Existing running normal tasks finish normally; the first preemptable wave waits for original task 3 to finish so the transition also respects the four-GPU limit. Seven pending tasks in the first array were canceled and replaced by preemptable tasks; these cancellations are deliberate, not failed experiments. Every wave continuation waits for both partition arrays. Preemptable jobs use Slurm requeue and the existing per-epoch checkpoint resume. Task assignments and replacement IDs are in `jobs.json`.

## Updated concurrency authorization

The user authorized four simultaneous GPUs on EACH partition (eight total). Both current arrays and all future waves use `%4` independently. The initial preemptable transition dependency was removed. Allocation remains 30 preemptable / 90 normal tasks; wave continuation still waits for both arrays before submitting the next wave. This supersedes earlier descriptions of a four-GPU combined cap and three-plus-one concurrency.

## Preemptable time segments

Preemptable tasks now request 90 minutes each. `train_cave_teasar_confidence_preemptable.sh` receives a batch-only USR1 signal two minutes before walltime and calls `scontrol requeue` for its array task. The trainer resumes from its atomically saved latest epoch checkpoint, including optimizer and RNG state, until the final report exists. Slurm preemption also uses requeue. Normal jobs retain their original limits. First-wave preemptable array 25534589 was canceled and replaced by 25534854 with this launcher; the continuation dependency was updated to the replacement.

## Batched loading rollout

The presynaptic trainer now uses `data/batched_presynaptic.py`. Its bulk dataset fetch groups reads by root, gathers ragged memberships and embeddings with vectorized indexing, and builds one packed PyG Batch directly. It preserves original window order, duplicate sampled windows, missing-embedding zeros/masks, edges and edge order, LPE, relative coordinates, targets and optional postsynaptic features. Augmentation policies retain the legacy draw order through a compatibility path. The sampler, training examples, architecture and optimizer are unchanged. Whole-cell loading is unchanged.

`batch_loading_benchmark_CAVE.json` and `batch_loading_benchmark_TEASAR.json` record warm-cache, identical-batch CPU benchmarks on 32 real training cells. CAVE: 9.08× single-worker / 6.84× four-worker loading speedup; TEASAR: 29.05× / 9.87×. These are loader-only measurements on 16 batches, not full-epoch GPU training speedups or cold-filesystem benchmarks. All compared real batch tensors match exactly. `scripts/benchmark_batched_presynaptic.py` reproduces the measurements.

The focused suite verifies ragged and repeated windows across roots, both edge-storage forms, missing embeddings, single-node mode, postsynaptic routing, augmentation RNG order, and identical mean/MLP/GT outputs and gradients. Combined loader/dense-dataset/config checks passed 24 tests and 9 subtests. `batch_loading_rollout.json` records hashes, benchmarks and checkpoints used to requeue the four active jobs. Queued and future jobs automatically load the new trainer; running jobs were requeued at the user's request and resume their saved epochs.
