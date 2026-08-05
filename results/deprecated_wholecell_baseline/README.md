# Deprecated -- whole-cell-averaged baseline, not the paper's methodology

**Deprecated 2026-08-05.** `baseline_depth2.json` here (79.6% accuracy, 72.1% balanced accuracy,
64.1% macro F1, real store data) was produced by a baseline that **averages a cell's
geodesic-mean-windowed node embeddings into one whole-cell vector before classifying once**.
That is not what the SegCLR paper's classifier does, and not what this lab's own
`aggregation_study/03_train_evaluate.py` replication does either -- both classify **each
25um-windowed point independently**, then get a cell-level answer by **majority-voting the
per-point predictions**. Confirmed from two independent sources:

- The original classifier gist
  (https://colab.research.google.com/gist/chinasaur/47b631677d099fa8059a7ce7c323222b):
  `train_embeddings.extend(e)` flattens every per-node windowed embedding in a cell into its
  own training example, labeled with the parent cell's type.
- `github.mit.edu/collina/segCLR_cell_classification`'s `aggregation_study/03_train_evaluate.py`:
  reports "per-point accuracy" and "cell-level majority-vote accuracy" explicitly, via
  `cell_majority_vote_accuracy()` -- never embedding averaging.

Averaging embeddings first, as this result did, changes what's actually being measured -- it
lets information from the whole cell blend into a single decision, rather than testing whether
a *local* 25um window is enough to classify correctly on its own (which is what "aggregation
radius" means in the paper). Not comparable to the corrected baseline in `results/` going
forward, and not a fair "baseline to beat" for the GNN comparison. The GNN's own scratch/
frozen/finetune numbers (`results/gnn_*.json`) are not invalidated by this -- they're a
different, deliberately whole-cell architecture per the Notion spec, and stand on their own --
but any narrative comparing them against this specific baseline number should be discarded.
