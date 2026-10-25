# Balanced-sampling reruns · 2026-10-09

Replay the three original five-class presynaptic experiment groups using
`class_balance=sample` (window sampling with replacement, weights proportional
to 1/sqrt(class count)) and unweighted classification loss. This requires fresh
training; existing loss-weighted checkpoints cannot be converted after training.

The 18 runs comprise:

- CAVE k10: mean pooling and Graph Transformer (2).
- Original TEASAR k10 with structural nodes: TEASAR Graph Transformer (1).
- Native TEASAR: mean pooling, pointwise MLP, Graph Transformer at scales
  4/8/16/32 (k65/k33/k17/k9), plus the existing mixed-cell scale32
  experiment (its saved setting is 16 cells per batch) (15).

`plan.json` records the original reports and their hashes, expected classes/test
window counts, config paths, and manifest hashes. `configs/` freezes each run's
original arguments, changing balancing and output location and explicitly
pinning its original manifest. `manifests/` contains those frozen manifests.
`preflight.json`, when present, records validated train/test counts.

Original loss-weighted reports/checkpoints remain untouched. New reports and
checkpoints go into a `balanced_sampling/` subdirectory beneath each original
result directory. Original seeds, classes, data, architecture, optimizer
settings, epoch counts (100 for original CAVE/TEASAR, 30 for native), and batch
settings are retained. Each job requests 16 CPUs and uses 15 data-loader workers. Resume applies only to the new sampling runs.

Validate original input coverage, labels, held-out window counts, and config
invariants before submission:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 segclr_db/.venv/bin/python \
  scripts/rerun_presynaptic_balanced_sampling.py --preflight
sbatch scripts/sbatch/rerun_presynaptic_balanced_sampling.sh
```

Training uses `mit_normal_gpu`, the `mit_amf_advanced_gpu` account/QOS, and
L40S GPUs. The job array allows at most four simultaneous GPU jobs. Each task
also checks its input/configuration provenance before starting training.
