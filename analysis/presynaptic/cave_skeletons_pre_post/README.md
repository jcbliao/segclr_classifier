# CAVE skeletons: pre + post analysis

Pointwise MLP and Graph Transformer, aggregating 10 raw presynaptic embeddings
from CAVE skeleton windows and concatenating the original 64D postsynaptic
embedding. Both models train on folds 0–4 for 30 epochs with the original
hyperparameters and confidence cutoff 0.7.

Classes are BasketFam; nonBasketInhibitory (BipFam, MartFam, NglFam); and
Excitatory (pyramidal, thalamocortical). Original cells and fold assignments are
preserved. A single three-output root classifier trains on these merged labels.
The class-balanced sampling policy uses the merged-class counts.

Input databases:
`/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_confidence_cave/k10/conf0.7/foldN`.
Post embeddings are aligned by synapse ID in a separate cache:
`/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_skeletons_pre_post/k10/conf0.7/postsynaptic`.
The preparation job requires post-inference coverage for every valid CAVE window
and excludes unresolved roots and empty post masks. It checks actual loader
inputs and both models' forward and backward passes on each fold and split.

Results are under
`results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/three_class/foldN/pre_post`.
The two model-named notebooks in `three_class/` compare folds after training completes.
Validation and checkpoint selection share the held-out split, as in the original
training; the notebook results are not an independent test estimate.

Submission records are in `three_class/submission.json`. For future submissions:

```bash
bash scripts/submit_cave_three_class_pre_post.sh
```

This replaces the native k17 three-class trainings in array 25175953; its k1
tasks 0, 3, 6, 9, and 12 continue separately.

## Notebook folders

- [three_class/](three_class/): Pointwise MLP and Graph Transformer five-fold analyses for BasketFam, nonBasketInhibitory, and Excitatory. Existing submission records and manifests remain here.
- [six_class/](six_class/): Pointwise MLP five-fold analysis for BasketFam, BipFam, MartFam, NglFam, pyramidal, and thalamocortical.

The six-class notebook reads saved metrics and epoch curves from
`results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class/foldN/pre_post`.
All five MLP folds have results. No six-class Graph Transformer results were
present when this notebook was created, so no corresponding Transformer analysis
is included. Each notebook shows window/cell metrics, training curves, class
recall/support, and row-normalized confusion matrices with raw counts.
