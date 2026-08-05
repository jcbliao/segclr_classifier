# Deprecated -- our own baseline reimplementation, superseded by the lab's actual infrastructure

**Deprecated 2026-08-05.** `baseline_depth2.json` here (71.6% cell-level accuracy, node-level
classify + majority vote, real store data) came from `baseline/mean_pool_classifier.py` +
`scripts/train_baseline.py` -- code written from scratch in this project to *approximate* the
SegCLR paper's node-level-classify-then-majority-vote methodology, after two rounds of fixing
it to match that methodology (see `results/deprecated_wholecell_baseline/README.md` for the
first bug this superseded).

That approximation is now itself superseded: `segCLR_cell_classification/` (cloned from
`github.mit.edu/collina/segCLR_cell_classification`) is the lab's own actual training
infrastructure for this exact task -- `DeepResNet` classifier, their `CellTypingDataset`/
`BaseTrainer`, their `cell_level_accuracy` majority-vote metric. Per explicit user direction,
the baseline going forward is trained through *that* infrastructure directly, not our
reimplementation of it. Do not compare new results against this file -- it measures a
different (home-grown) classifier architecture and training procedure, even though the
underlying methodology (per-point classify, majority vote) is the same idea.

`baseline/mean_pool_classifier.py`/`scripts/train_baseline.py` are kept in the repo for
reference only, not as an active baseline.
