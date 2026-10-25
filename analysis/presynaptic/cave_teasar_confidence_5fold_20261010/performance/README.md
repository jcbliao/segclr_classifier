# Training performance audit — 2026-10-10

The trainer now uses FP16 autocast with GradScaler when `--amp` is enabled, as requested. Last checkpoints save/restore scaler state; older BF16 checkpoints resume with a fresh scaler. Existing runs keep their earlier BF16 epochs and best checkpoint. Completed/reused reports are unchanged; this rollout is not a new precision-controlled experiment.

Changes: batched root reads/tensor construction; skip unused geometry for mean/linear/pointwise MLP; pinned nonblocking transfers; prepare hierarchy loss routes on CPU; use CPU-known batch dimensions; accumulate loss on GPU with one epoch-level read; static hierarchy evaluation; collect evaluation predictions with one final device transfer; vectorized class voting and sampling. Sample draws/order, weighting, Adam optimizer, model architecture, losses and selection metric remain unchanged. Checkpoint frequency is retained because writes were small.

## GPU measurements

L40S, 16 CPUs/15 loader workers; identical 64 timed batches after 8 warmup batches, 64 real roots per skeleton family. Baseline already includes the earlier batched loader. Both measurements use FP16, GradScaler and Adam. These are short warm-cache measurements, not full-epoch speedups. Shared filesystem/worker scheduling introduces variability. In particular, TEASAR GT did not improve in this short benchmark. Full-epoch telemetry is authoritative for deployment.

| Skeleton | Model | Baseline training s | Updated training s | Speed ratio | Synchronization waits / four steps |
|---|---|---:|---:|---:|---:|
| CAVE | mean | 0.897 | 0.842 | 1.07× | 72 → 4 |
| CAVE | pointwise_mlp | 1.250 | 1.043 | 1.20× | 72 → 4 |
| CAVE | graph_transformer | 1.345 | 1.057 | 1.27× | 128 → 36 |
| TEASAR | mean | 2.161 | 1.928 | 1.12× | 56 → 4 |
| TEASAR | pointwise_mlp | 2.087 | 2.024 | 1.03× | 56 → 4 |
| TEASAR | graph_transformer | 2.078 | 2.228 | 0.93× | 112 → 36 |

Voting 450,000 points / 500 roots: 0.1377 → 0.0367 seconds. Sampling took roughly 0.03–0.05 seconds after optimization, versus 0.06 seconds. Evaluation was measured over complete fixed loader iterations, with alternating old/new order. Gains were small and variable; see the JSON rather than assuming a large evaluation speedup.

## Validation and rollout

34 CPU performance/batching tests passed; 12 configuration/dense tests and 9 subtests passed. The GPU allocation passed 25 performance tests, including CUDA losses, gradients and Adam-step equivalence. Real FP16 mean/pointwise MLP/GT training passed, with finite saved weights, saved scaler state and a two-epoch mean resume check. Independent mixed-precision GPU reductions can vary slightly near decision boundaries; same-hidden hierarchy routing matched exactly, and independent FP16 evaluation disagreements were below 0.1% (also present across repeated evaluations). Exact CPU outputs/gradients, batch order, true labels and root IDs were checked.

`rollout.json` records the active runs requeued at saved epochs. The held training job was released. Normal concurrency returns to four after the testing allocation finishes; preemptable remains four. New and resumed jobs use FP16 and updated code. Every new epoch appends `performance.jsonl` beside the checkpoint with training, host loader wait, evaluation, metrics, checkpoint/logging, window count, precision and gradient scale. Host loader wait is not GPU idle time.

Artifacts: `gpu_profile_baseline.json`, `gpu_profile_optimized.json`, `fp16_validation.json`, `rollout.json`, and smoke logs. Frozen reference loader/sampler modules reproduce the before implementation. `gpu_profile_initial_fp16.json` is an earlier candidate using partial-iterator evaluation timings and is not the final evaluation benchmark.


## Later GPU-resident rollout

See [resident_loading.md](resident_loading.md) for the subsequent packed/GPU-resident pipeline, substantially larger speedups, exact-data checks, and deployment records. Earlier timings above describe the previous loader.
