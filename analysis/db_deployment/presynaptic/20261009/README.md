# Presynaptic deployment analysis · 2026-10-09

- `test_analysis.ipynb`: selected-class and true-class probability histograms, accuracy by probability bin, and a 6×6 matrix of probability histograms by true/predicted pair.
- `unlabeled_analysis.ipynb`: selected-class histogram, probability versus actual embedding count, six class-specific curves, and support counts.

Both default to the six-class `cave_n10` pointwise MLP, five folds. Set `FAMILY = 'single_pre_post'` after preparing its cache to analyze the alternative. Joint probabilities are decoded through the hierarchy, and the selected class is the original top-down decision.

## Build cached data

From the repository root:

```bash
sbatch analysis/db_deployment/presynaptic/20261009/build_cache.sh
# Optional alternative model:
sbatch analysis/db_deployment/presynaptic/20261009/build_cache.sh --family single_pre_post
```

Five CPU tasks run concurrently with parallel data loading and DuckDB reads under `mit_amf_advanced_cpu`. Completed caches are reused when provenance matches. Cache location:

`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/db_deployment/presynaptic/20261009/{family}/`

Test caches contain all held-out window joint probabilities, targets, predictions, root IDs, and synapse IDs. They are reconstructed in FP32 from best checkpoints and original dataset/split settings; accuracy is checked against the saved mixed-precision evaluation (tolerance 0.002 for precision-induced decision differences). The training pipeline selected checkpoints using these same held-out splits.

Unlabeled caches contain exact histogram counts and sums/squared sums by actual embedding count, computed from saved deployment predictions. Each fold retains its own hard presynaptic axon filter. No synapses are deduplicated for these summaries. Class-specific plots filter each fold to synapses selecting that class, then average its selected-class joint probability. Class-specific support counts are retained for confidence intervals.

Shading uses nested 50%, 80%, 95% point-level normal intervals for the mean. These intervals do not account for clustering by presynaptic root or shared fragment, and are not estimates of classification accuracy. Histogram composites pool fold records rather than averaging probabilities across folds. Nothing here writes to shared segclr_db.
