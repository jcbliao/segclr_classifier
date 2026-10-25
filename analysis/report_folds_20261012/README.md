# Five-fold report experiments — 2026-10-12

**204 new training runs**, plus 226 reused runs across 86 model configurations.

The original request was 189. Switching the density sweep to morpho requires replacing its 12 historical fold-0 runs (+12). Applying MICrONS-only validation selection to both V1DD arms adds three baseline fold-0 reruns (+3).

| Experiment | New training runs |
|---|---:|
| Whole-cell linear versus ResNet | 32 |
| Morpho TEASAR density sweep | 60 |
| Historical CAVE versus original mesh TEASAR | 12 |
| V1DD addition | 30 |
| TEASAR neighborhood pre/post | 28 |
| Single-embedding pre/post | 32 |
| CAVE six-class pre+post | 10 |

## Models and folds

- Density: six morpho classes, confidence ≥0.7, bin32/k9, bin16/k17, bin8/k33, bin4/k65, mean/pointwise MLP/full GT, five folds. All 60 runs are new. The same morpho root assignments are used at every density.
- Whole-cell and historical mesh experiments preserve their original fold-0 test roots. Remaining roots are stratified by fine label across folds 1–4; each root is held out once. Presynaptic morpho experiments reuse their existing five-fold manifests.
- V1DD: both arms use MICrONS validation-window macro F1 to choose epochs. V1DD training examples remain in the addition arm; both domains are retained in final reports. Original fold-0 results are preserved; corrected results live under `cave_v1dd_addition/microns_selection/`.
- Existing model-specific epoch budgets and training hyperparameters are retained, with current FP16, optimized loading, and CUDA-graph defaults. No aggregator is frozen. Training uses balanced sampling.
- The 13 added thalamocortical roots had embeddings but lacked density window caches. Their 52 multiscale cell records were built from cached geometry, soma-cut centers, and synapses, using the same full-resolution-center policy as the existing density data. No new embedding inference or remote queries were needed.

## Execution

- Four normal workers (array 25641194) and four preemptable workers (array 25641195), each with 16 CPUs / one L40S / 64 GB. About one quarter of tasks are assigned to preemptable.
- Preemptable workers use 30-minute allocations and requeue two minutes before the time limit and resume their owned task from its latest epoch checkpoint. Normal workers use six-hour segments. Arrays consume the frozen queue rather than submitting hundreds of Slurm tasks at once.
- `queue.json` records each task’s worker, state, and failures. `logs/report_fold_taskNNN.log` contains its training output. Failed tasks are retried once; the final audit reports any missing outputs.
- Additional inference covers structure ablations for folds 1–4 and augmentation propagation for all five clean-trained GT folds. These are separate from the 204 training runs.
- Final notebook rendering waits for both training arrays and follow-up inference. `final_status.json` records missing reports, missing inference caches, and rendering errors explicitly.

## Analysis

Every listed entry point and its `*_all_folds.ipynb` sibling read the frozen catalog. Original notebook versions are archived in `legacy_notebooks/`. Plot families include window/cell metrics, learning curves, paired changes, class recall/precision/support, confidence/density trends, cohort composition, normalized confusion matrices, structure statistics, and augmentation propagation.

Fold means have equal weights. Pointwise 95% Student-t intervals use five folds (four degrees of freedom); paired differences are formed within folds first. Confusion matrices average row-normalized fold matrices and annotate each cell with its mean and interval. Unsupported class rows are marked, and incomplete groups do not generate partial five-fold summaries. Random perturbation seeds are averaged within folds before calculating fold intervals. Training sets overlap, so these intervals describe fold variation rather than independent-replicate uncertainty.

CAVE pre+post pages compare the registered six-class post-input models. The optional 30 additional matched pre-only / presynaptic-mean control runs discussed in the audit are outside this queue. Historical mixed-batch pilots remain archived rather than being included as if five folds had been trained.

## Notebook index

- [linear_vs_resnet_all_folds.ipynb](../../analysis/all_windows/linear_vs_resnet_all_folds.ipynb) — Whole-cell linear versus ResNet
- [architecture_comparison_all_folds.ipynb](../../analysis/all_windows/architecture_comparison_all_folds.ipynb) — Whole-cell linear versus ResNet
- [skeleton_comparison_native_all_folds.ipynb](../../analysis/presynaptic/new_skeletons_native/skeleton_comparison_native_all_folds.ipynb) — Morpho TEASAR density sweep
- [skeleton_comparison_all_folds.ipynb](../../analysis/presynaptic/skeleton_comparison/skeleton_comparison_all_folds.ipynb) — Historical CAVE versus original mesh TEASAR
- [embedding_augmentation_comparison_all_folds.ipynb](../../analysis/presynaptic/cave_v1dd_addition/embedding_augmentation_comparison_all_folds.ipynb) — V1DD addition
- [pre_post_comparison_all_folds.ipynb](../../analysis/presynaptic/native_skeletons_pre_post/pre_post_comparison_all_folds.ipynb) — TEASAR neighborhood pre/post
- [pre_post_comparison_all_folds.ipynb](../../analysis/presynaptic/native_single_pre_post/pre_post_comparison_all_folds.ipynb) — Single-embedding pre/post
- [pre_post_comparison_all_folds.ipynb](../../analysis/presynaptic/cave_skeletons_pre_post/six_class/pre_post_comparison_all_folds.ipynb) — CAVE six-class pre+post
- [embedding_augmentation_comparison_all_folds.ipynb](../../analysis/presynaptic/cave_embedding_augmentation_conf0.7/embedding_augmentation_comparison_all_folds.ipynb) — EM augmentation
- [morpho_confidence_comparison_all_folds.ipynb](../../analysis/presynaptic/morpho_confidence_cave/morpho_confidence_comparison_all_folds.ipynb) — Morpho CAVE/TEASAR confidence sweep
- [morpho_confidence_comparison_all_folds.ipynb](../../analysis/presynaptic/morpho_confidence_native/morpho_confidence_comparison_all_folds.ipynb) — Morpho CAVE/TEASAR confidence sweep
- [comparison_all_folds.ipynb](../../analysis/presynaptic/cave_teasar_confidence_5fold_20261010/comparison_all_folds.ipynb) — Morpho CAVE/TEASAR confidence sweep
- [paired_comparison_all_folds.ipynb](../../analysis/presynaptic/axon_vs_whole_cell_morpho6/paired_comparison_all_folds.ipynb) — Paired axon versus axon+dendrite
- [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_5folds_all_folds.ipynb](../../analysis/presynaptic/native_skeletons_pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n17_mixed16_sampled_pre_post_5folds_all_folds.ipynb) — TEASAR neighborhood pre/post
- [gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_5folds_all_folds.ipynb](../../analysis/presynaptic/native_skeletons_pre_post/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n17_mixed16_sampled_pre_post_5folds_all_folds.ipynb) — TEASAR neighborhood pre/post
- [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_5folds_all_folds.ipynb](../../analysis/presynaptic/native_single_pre_post/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n1_mixed16_sampled_pre_post_5folds_all_folds.ipynb) — Single-embedding pre/post
- [gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_six_class_5folds_all_folds.ipynb](../../analysis/presynaptic/cave_skeletons_pre_post/six_class/gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_pre_post_six_class_5folds_all_folds.ipynb) — CAVE six-class pre+post
- [mean_pre_post_all_folds.ipynb](../../analysis/presynaptic/cave_skeletons_pre_post/six_class/mean_pre_post_all_folds.ipynb) — CAVE six-class pre+post
- [gt_pre_post_all_folds.ipynb](../../analysis/presynaptic/cave_skeletons_pre_post/six_class/gt_pre_post_all_folds.ipynb) — CAVE six-class pre+post
- [structure_ablation_all_folds.ipynb](../../analysis/all_windows/structure_ablation_all_folds.ipynb) — Whole-cell inference interventions

## Cache locations

- Source windows/embeddings: paths frozen in `configs/` and `manifests/`; existing files are reused.
- Density additions: `/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/report_folds_20261012/density/`.
- Whole-cell prediction caches: shared scratch under `report_folds_20261012/predictions/taskNNN/`, preventing collisions between run names in different experiments.
- Structure packed windows: shared scratch under `whole_cell_structure_n10_20261011_foldN/`.
- Final metrics and checkpoints: per-run locations in `plan.json`; no segclr_db prediction writes are performed.

Validation: all 204 JSON configs pass the training parser; 37,496 input paths were checked. GPU validation passed 23 tests before launch. Additional tests cover fold coverage, domain selection, resumable queue ownership, and confidence intervals.


## Persistent CPU packing cache

Packed arrays live at `/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/packed_windows/`.
`packing_plan.json` maps all 204 training tasks to 114 reusable dataset caches.
CPU array **25643707** has no array concurrency cap and builds as many caches concurrently as Slurm permits, using 16 CPUs and 128 GiB RAM per job on `mit_normal` / `mit_amf_advanced_cpu`.
Whole-cell builders use 15 forked CPU processes writing disjoint memory-mapped array sections; presynaptic builders parallelize cell packing with 15 threads, alongside the parallel array jobs. Both display tqdm progress.
Cache keys include input file sizes and modification times, manifest contents, input feature settings, and dataset/packing code hashes. Model heads and compatible aggregators share caches.
A completion marker is published atomically after all arrays are written, with a filesystem lock preventing duplicate builds. GPU jobs load arrays via memory mapping and transfer features to VRAM under the existing free-memory reserve check.
The eight training workers were held during the switch; their CPU builders automatically release them when their current cache is ready. Workers only claim tasks with completed caches and resume saved epoch checkpoints. Existing downstream analysis dependencies retain the original GPU job IDs.
Stochastic training augmentation is never cached as fixed augmented samples. Packed cache identity must be rebuilt if source files or dataset code change.

Training queue allocation updated to **102 normal + 102 preemptable runs**, retaining four workers on each partition and ownership of all current runs. Preemptable allocations remain 30 minutes.

Packing array throttle removed (`ArrayTaskThrottle=0`). CPU QoS limits are 512 CPUs and 2 TiB RAM per user; at 16 CPUs / 128 GiB per packing job, memory permits up to 16 simultaneous builders if the full allowance is available.

Four historical TEASAR graph caches (builder indices 52–55, training tasks 100–103) exceeded the 44 GiB CPU packing estimate limit (71–73 GiB). Retry array **25658415** requests 256 GiB host RAM and a 96 GiB CPU packing allowance, with no array concurrency throttle. This does not change cache identities or the GPU free-memory residency check. Oversized caches can remain CPU-mapped and stream batches to the GPU.

Oversized historical TEASAR GT folds 1–4 (tasks 100–103) are reserved for H200 array **25658709** on `mit_normal_gpu` / `mit_amf_advanced_gpu`, after successful CPU retry array 25658415. Each requests one H200, 16 CPUs, 128 GiB host RAM, six-hour resumable segments. Queue ownership prevents the L40S worker pools from claiming these tasks. Downstream analyses and final notebook rendering now wait for this array as well. This hardware exception supersedes the even partition split for these four tasks.

Remaining-work rebalance: moved 19 unstarted preemptable tasks to the four normal workers, with explicit queue ownership preventing stale preemptable workers from claiming them. At rebalance, 23 remaining runs were assigned to normal (including four H200 folds), and 23 to preemptable (including four active folds). `execution=assigned_pending` denotes reserved work awaiting its worker, not an actively training model. Normal workers were requeued to reload the plan; active preemptable training was preserved.

User superseded the even split: all remaining training now uses `mit_normal_gpu`. Preemptable worker array 25641195 was cancelled; unfinished runs/checkpoints were transferred to normal workers. H200 array 25658709 remains responsible for tasks 100–103. The four normal L40S workers retain six-hour resumable allocations.
