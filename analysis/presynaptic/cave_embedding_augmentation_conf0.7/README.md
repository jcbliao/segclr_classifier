# Analysis: CAVE embedding augmentation comparison

This directory contains summaries for the four-model confidence 0.7 fold 0
experiment. Run `scripts/summarize_cave_embedding_augmentation.py` after all
four training jobs complete to write `model_comparison.csv` here.

The held-out set is clean for every condition. The comparison therefore asks
whether training-time image-derived embedding variation improves clean-test
generalization.
