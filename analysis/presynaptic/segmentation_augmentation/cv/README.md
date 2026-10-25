# Five-fold cumulative mask augmentation comparison

Nine conditions, five existing folds: 45 runs of the current six-class CAVE
K10 pointwise MLP pre/post classifier. Configurations clone each fold's saved
clean baseline hyperparameters (30 epochs, seed 0, class-balanced sampling,
mixed-cell batches), with independent result directories. Post inputs stay
unchanged. Held-out evaluation uses the original clean embeddings.

Each observed node independently samples uniformly at every presentation:

| Maximum layers | Shrink only | Expand only | Both |
|---|---|---|---|
| 1 | clean, shrink1 | clean, expand1 | clean, shrink1, expand1 |
| 2 | clean, shrink1–2 | clean, expand1–2 | clean, shrink1–2, expand1–2 |
| 3 | clean, shrink1–3 | clean, expand1–3 | clean, shrink1–3, expand1–3 |

Clean appears exactly once in each pool. There is no fixed 50% clean rate.
The loader selects columns from the shared seven-view cache; it does not
duplicate the embedding database for each condition. Clean columns use the
original source embeddings to match the existing clean baselines.

`experiment.json` records conditions, pool sizes, baseline and manifest hashes,
task/config mappings and embedding inputs. `configs/` contains the 45 resolved
training configurations. `jobs.json` records the submission chain:

- Embedding generation/finalization dependency: 26117504.
- Preflight: 26117681, checking complete caches, all-fold geometry identities,
  class-fold manifests, and disjoint held-out assignment coverage.
- Training arrays: 26117682 (tasks 0–29, four concurrent preemptable GPUs) and
  26117683 (tasks 30–44, two normal GPUs).
- Summary: 26117684, after all 45 training runs succeed.

Task mapping: `fold = task // 9`; condition is `task % 9` in the order shrink1,
expand1, both1, shrink2, expand2, both2, shrink3, expand3, both3. Original split
assignments are preserved. Augmentations may be generated for every cell, but
each fold's training loader only augments that fold's training cells.

The summary produces `fold_metrics.csv` and `condition_comparison.csv` with
cell/window accuracy, balanced accuracy and macro F1, fold means and standard
deviations, and paired differences from each fold's saved clean baseline.
The existing workflow uses held-out folds for checkpoint selection too; these
metrics follow that established protocol, not an independent final test set.

## Current recovery chain

Earlier mixed-partition/dependency-cancelled CV jobs have been replaced.
Embedding finalization 26130457 feeds preflight 26130458, then array
26130459 (all 45 runs, up to four GPUs, mit_normal_gpu only), then summary
26130460. node3006 is excluded after repeated CUDA initialization failures.
See jobs.json for authoritative current submission metadata.

## Verified fast augmentation training

The original augmentation path explicitly bypassed packed caching and GPU
residency despite optimized-loading being enabled. Pending tasks 32–44 were
held while this was corrected, then released after validation. A canonical
seven-view bank stores each distinct node once; packed window occurrence IDs
index it. Independent uniform per-node view draws run inside CUDA graph
replays, with the same cumulative pools and original clean vectors.

CPU caches are shared by all conditions within a fold. Resident training is
required for these policies when optimized loading is enabled; there is no
silent fallback to streaming. Existing completed results are retained.

The real fold3 three-epoch GPU pilot (26168301) passed all 13 bank tests,
including fresh draws across graph replays. Warm training time was 1.294 s
per epoch versus 73.563 s for the previous streaming path (56.85x). This
comparison excludes startup, evaluation and checkpoint overhead. See
fast_training_benchmark.json and fast_training_jobs.json.
