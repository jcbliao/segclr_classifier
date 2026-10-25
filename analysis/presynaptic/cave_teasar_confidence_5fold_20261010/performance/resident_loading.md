# Packed and GPU-resident training input

The same selected training/test windows, node order, embeddings (float32 storage), missing-node masks, geometry, labels, root IDs, centers and sampled draws are preserved. No extra aggregation, filtering, quantization or batch-size changes were introduced. The existing sampler drives every loader.

Each job packs selected windows once using up to eight CPU threads. CAVE packing took 37.6 s / 4.13 GiB; TEASAR took 66.7 s / 9.67 GiB with geometry enabled. Mean/MLP omit unused geometry. Window features are uploaded once when GPU memory permits, leaving at least 8 GiB for training. Batches gather features directly on the GPU; CPU metadata is pinned and transferred asynchronously. A bounded two-batch producer queue and a separate CUDA stream overlap batch preparation/transfers with the current model step. Packed CPU input remains a fallback when GPU residency cannot fit. Stochastic embedding augmentation retains the original data path.

CUDA training uses fused Adam, with the same lr/weight decay/objective and FP16 GradScaler. CUDA tests checked update equivalence and overflow skipping. Resume migrates old CPU Adam step counters to the GPU and restores scaler state. Checkpoint selection/frequency remain unchanged.

## Identical-batch benchmark

L40S, 16 CPU allocation, full source training cohort (~2,000 roots across train/test), actual saved architecture and ResNet-head configurations. Same 136 batches in each mode; 8 warmup / 128 timed. Baseline is the previous batched loader with 15 workers and pinned transfers; new is GPU-resident input, prefetch and fused Adam. Packing/startup excluded here and stated above. Short benchmark speedups are not promises for whole jobs.

| Skeleton | Model | Before s | After s | Speedup | Host batch wait | Model-step CUDA timeline / wall |
|---|---|---:|---:|---:|---:|---:|
| CAVE | mean | 2.626 | 0.325 | 8.08× | 12.0% | 88.6% |
| CAVE | pointwise_mlp | 2.875 | 0.360 | 7.98× | 10.7% | 89.7% |
| CAVE | graph_transformer | 2.806 | 1.165 | 2.41× | 3.1% | 100.0% |
| TEASAR | mean | 5.375 | 0.329 | 16.36× | 11.5% | 89.0% |
| TEASAR | pointwise_mlp | 5.381 | 0.360 | 14.93× | 10.8% | 89.6% |
| TEASAR | graph_transformer | 5.346 | 1.312 | 4.07× | 2.8% | 100.0% |

CUDA event durations measure the model/optimizer step timeline, including launch gaps; they are not SM utilization or kernel-only busy time. Host wait may overlap already queued GPU computation. The result establishes that repeated decoding, CPU gathering and PCIe feature transfers no longer dominate this benchmark; it does not claim 100% hardware utilization.

## Validation and deployment

62 CPU tests plus 9 subtests passed. GPU validation passed all 43 packed/performance tests, including exact tensor parity, repeated draws, missing embeddings, precomputed/derived topology, single/pre+post variants, stream handoff/cleanup, model gradients and fused Adam. Full FP16 trainer smoke tests passed for mean/MLP/GT. Mean checkpoint resume deliberately simulated the old non-fused optimizer and CPU step counters and passed.

`resident_rollout.json` records active jobs requeued from saved checkpoints. The testing job completed, held task 21 was released, and concurrency was restored to four normal plus four preemptable. Every new epoch writes resident/packed/prefetch/fused flags and `model_step_cuda_seconds` in its `performance.jsonl` beside existing timing fields. Existing completed reports remain unchanged.

Sources: `data/packed_presynaptic.py`, `gnn/cuda_prefetch.py`, `scripts/train_gnn.py`. Benchmark: `scripts/profile_resident_loading.py`, `resident_loading_profile.json`. GPU jobs: 25539705 (benchmark) and 25539814 (43 tests and trainer/resume checks).


## Final training path: replay and parallel epoch plans

The warmed GPU training loop now replays the full forward, backward, FP16 scaler and fused Adam step. Two alternating buffers and a separate transfer stream overlap copying the next batch's window IDs with current GPU work. Features and graph topology stay resident. Buffers round capacity to 128 windows; padding contributes neither embeddings, node counts nor loss. Original real windows, repeated draws, labels, batch boundaries and optimizer-step counts are preserved. CUDA graphs share a memory pool and cache by capacity, active hierarchy heads and buffer slot. Unsupported stochastic configurations keep the existing fallback.

All seeded epoch plans are computed once, in up to eight CPU worker processes, before training. Tests compare every batch against the original sampler for both ordinary and mixed-cell sampling, including the parallel process path. This removes the CPU sampling pause before each epoch. Packing, plan preparation and initial graph capture remain one-time startup costs.

L40S profiler measurements below are kernel interval union divided by the warmed training GPU timeline, over 40 batches. They exclude initial capture/startup and evaluation. They measure kernel activity, not SM occupancy or whole-job GPU utilization. Full source cohorts and saved model configurations were used.

| Skeleton | Model | Resident eager kernel activity | Full-step replay kernel activity |
|---|---|---:|---:|
| CAVE | mean | 14.3% | 93.8% |
| CAVE | pointwise MLP | 13.2% | 93.4% |
| CAVE | graph transformer | 30.3% | 97.7% |
| TEASAR | mean | 13.6% | 92.7% |
| TEASAR | pointwise MLP | 15.6% | 93.4% |
| TEASAR | graph transformer | 61.0% | 98.7% |

Evidence: `bucket_graph_profile_CAVE.json`, `bucket_graph_profile_TEASAR.json`; profiler/GPU-test job 25541402. Ordinary compilation and exact-shape capture were tested and rejected because changing batch shapes caused excessive overhead; their JSON artifacts remain for audit.

Final GPU validation before the epoch-plan addition passed 55 tests, covering padded-buffer model/update parity for all three architectures, repeated updates, missing nodes, topology, transfers, GradScaler growth and overflow. Real FP16 training/checkpoint/resume also passed (`final_validation.log`). The epoch-plan addition passed 32 CPU tests with three CUDA tests skipped on the login node. Full sweep tasks 44 and 45 exercise parallel planning on the actual cohorts. Noninteractive batch progress rendering is disabled; epoch logs remain. CUDA step timing samples one step in 32 and is explicitly marked estimated in `performance.jsonl`.

Normal and preemptable queues retain independent four-job limits. Preemptable waves depend on the preceding preemptable wave; normal waves no longer wait for unscheduled preemptable work. Duplicate-wave submission is guarded and tested. No testing holds remain.

Full-cohort validation after parallel planning (`final_full_cohort_validation.json`): TEASAR mean processes 1,741,826 real sampled windows per epoch in approximately 0.88 s, with roughly 0.0004 s of host batch waiting. CAVE transformer also successfully trains with parallel precomputed plans. Plan preparation took 31.8 s for all 30 TEASAR epochs (199.3 MiB), once at startup. These are measured training times; evaluation and checkpointing are separate fields.
