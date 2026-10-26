# Reporting classification results

User preference, established 2026-10-26:

- Omit ordinary accuracy from new user-facing summaries, tables, and plots.
- Report balanced accuracy together with per-class precision. Macro precision
  can accompany aggregate comparisons, but does not replace per-class precision.
- Macro F1 is the preferred standalone metric when reporting a single score.
- Apply this convention to window/cell results, each hierarchy level, and
  conditional local-head diagnostics. Label the evaluation population clearly.
- State the zero-denominator convention for precision and include class support
  or predicted counts so never-predicted classes are visible.
- Existing training artifacts may retain accuracy fields for compatibility;
  changing presentation does not require retraining or rewriting frozen results.

# Publishing predictions

- Publish independent folds concurrently. Default to as many workers as the
  allocated CPU and memory support, up to the number of independent folds;
  do not impose an arbitrary two-worker cap.
- Register shared model metadata once before launching workers. Preserve
  per-run retry safety, exact label/logit verification, and a final receipt
  written only after every fold succeeds.
