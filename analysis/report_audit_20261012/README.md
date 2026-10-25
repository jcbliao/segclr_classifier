# Report five-fold audit — 2026-10-12

Report outline: canonical n10 whole-cell heads, historical balanced-sampling skeleton and density comparisons, six-class pre/post models including matched-width controls, all confidence thresholds, GT augmentation, V1DD, paired axon/dendrite experiment. Excludes historical alternative architectures/window sizes and three-class prediction deployment experiments.

| Experiment | Completed / expected | Missing fold runs |
|---|---:|---:|
| Whole-cell linear versus ResNet | 8 / 40 | 32 |
| Historical TEASAR density sweep | 12 / 60 | 48 |
| Historical CAVE versus original mesh TEASAR | 3 / 15 | 12 |
| V1DD addition | 6 / 30 | 24 |
| TEASAR neighborhood pre/post | 17 / 45 | 28 |
| Single-embedding pre/post | 13 / 45 | 32 |
| CAVE six-class pre+post | 5 / 15 | 10 |
| Morpho CAVE/TEASAR confidence sweep | 150 / 150 | 0 |
| EM augmentation | 20 / 20 | 0 |
| Paired axon versus axon+dendrite | 10 / 10 | 0 |

Missing: **186 fold runs**. Add three V1DD fold-0 reruns if changing epoch selection to MICrONS-only (189 total). The 12 other V1DD-with-addition missing folds must also use the corrected selection rule.

Optional: 30 additional CAVE runs for three architectures × five folds × matched pre-only / presynaptic-mean concatenation control. Existing confidence-sweep pre-only scores need not cover the same post-eligible windows.

TEASAR neighborhood comparison includes its own matched pre-only controls; confidence-sweep baselines are not substituted. Original density experiments use five historical neuronal classes, whereas the morpho sweep uses six; do not pool them.

Whole-cell inference perturbations currently use fold 0. Applying them to folds 1–4 requires inference after the whole-cell training completion above, not extra training beyond that count.

Counts represent missing final reports with window test metrics; see fold_audit.json for exact report paths and missing fold indices. No jobs were launched by this audit.
