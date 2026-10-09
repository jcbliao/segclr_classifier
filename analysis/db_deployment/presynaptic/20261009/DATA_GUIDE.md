# Presynaptic deployment data guide

This guide describes the data used by `test_analysis.ipynb` and
`unlabeled_analysis.ipynb` in this directory. Paths and cache schemas were checked
on 2026-10-09. Ordinary notebook runs read cached files; they do not run inference
or query/write the shared cell-type database (`segclr_db`).

## Storage locations

### Notebook analysis caches

```text
/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/
└── db_deployment/presynaptic/20261009/
    └── cave_n10/
        ├── test_fold0.npz … test_fold4.npz
        └── unlabeled_fold0.npz … unlabeled_fold4.npz
```

These compressed NumPy files are on shared storage. Currently only `cave_n10`
analysis caches exist. The notebooks and cache builder also support
`single_pre_post`, but its analysis caches must be built before selecting it.
Both notebook analyses currently use the **six-class classifier**.

### Full deployment predictions

```text
/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/
└── incoming_presynaptic_fragments_k10_collina_v2/
    ├── plan.json                         # Fragment construction and source provenance
    ├── parts/                            # Source fragments and their embedding inputs
    ├── embedding_count_distribution.csv # Source embedding-count distribution
    └── cell_types/
        ├── parts/<rank>/<batch>.parquet   # Full cell-type prediction records
        ├── predictions.duckdb            # Views over the Parquet files
        ├── plan.json                     # Frozen model/checkpoint/class metadata
        ├── summary.json                  # Prediction counts and validation summary
        ├── completion_audit.json         # Completion and data integrity audit
        └── real_data_parity_audit.json    # Independent model/output parity checks
```

The full deployment dataset has **8,125,826 distinct synapse rows**. It stores
predictions for two model families × two class sets × five folds: 20 model/fold
combinations. There are **160,190,532 included model/fold predictions**, not that
many distinct synapses. Missing selected-node embeddings exclude 394 synapses;
additional exclusions depend on each fold's presynaptic axon classification.

`predictions.duckdb` holds views referencing the Parquet files, so the DuckDB file
alone is not a self-contained copy of the predictions. These deployment
predictions have **not been imported into the shared segclr_db**.

## Common class and probability conventions

The six-class cache column order is:

| Index | Class |
|---:|---|
| 0 | BasketFam |
| 1 | BipFam |
| 2 | MartFam |
| 3 | NglFam |
| 4 | pyramidal |
| 5 | thalamocortical |

Always read `classes` rather than assuming this ordering in new code.
Probabilities are joint leaf probabilities decoded through the classifier's
hierarchy. They sum to one across these six classes. Raw logits in the full
prediction files belong to conditional local heads; applying one softmax to all
six logits does not recover the joint probabilities.

The **selected class** is the model's saved top-down hierarchy decision, which
can differ from the largest joint leaf probability. Selected-class probability
is the joint probability at that selected index. True-class probability uses
the ground-truth index and is available only in the labeled test caches.

## Test cache: individual labeled windows

Each `test_fold{fold}.npz` contains predictions for that fold's held-out windows,
reconstructed in FP32 from its saved best checkpoint and original dataset/split
settings. These use the training dataset's eligibility and window deduplication
rules, rather than deployment synapse counting rules.

| Array | Shape | Meaning |
|---|---|---|
| `probabilities` | `(N, 6)` | Float32 joint class probabilities |
| `prediction` | `(N,)` | Selected class index |
| `target` | `(N,)` | Ground-truth class index |
| `root_id` | `(N,)` | Labeled presynaptic source root |
| `synapse_id` | `(N,)` | Synapse associated with the retained test window |
| `classes` | `(6,)` | Class names in probability-column order |
| `provenance` | scalar string | JSON: format version 1, checkpoint SHA-256, manifest SHA-256, original run arguments |

| Fold | Test windows |
|---:|---:|
| 0 | 293,953 |
| 1 | 303,352 |
| 2 | 288,259 |
| 3 | 297,192 |
| 4 | 290,232 |

The original training pipeline used these held-out splits for checkpoint
selection too; these are not an independent final test evaluation. Reconstructed
FP32 accuracy is checked against saved mixed-precision metrics with an absolute
accuracy tolerance of 0.002. Small precision-induced decision differences are
possible. Cache reuse checks checkpoint, manifest, and argument provenance.

### Read individual test predictions

```python
from pathlib import Path
import numpy as np

cache = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/'
             'db_deployment/presynaptic/20261009/cave_n10')
with np.load(cache / 'test_fold0.npz', allow_pickle=False) as z:
    p = z['probabilities'].copy()
    predicted = z['prediction'].copy()
    true = z['target'].copy()
    classes = z['classes'].copy()

rows = np.arange(len(predicted))
selected_probability = p[rows, predicted]
true_probability = p[rows, true]
correct = predicted == true
```

## Unlabeled cache: exact summaries of saved deployment predictions

Each `unlabeled_fold{fold}.npz` summarizes the saved six-class predictions for
that fold after its own hard presynaptic axon filter. Distinct synapses sharing a
fragment remain separate contributions. No true labels or individual prediction
rows are stored in this compact cache.

| Array | Shape | Meaning |
|---|---|---|
| `hist` | `(50,)` | Selected-class probability histogram counts over all included synapses |
| `bins` | `(51,)` | Histogram edges, evenly spaced from 0 to 1 |
| `n` | `(10,)` | Included synapse counts for actual embedding counts 1–10 |
| `sums` | `(10, 7)` | Probability sums by embedding count: column 0 is selected-class probability; columns 1–6 are P(class) over all included synapses |
| `squares` | `(10, 7)` | Corresponding sums of squared probabilities |
| `selected_class_n` | `(10, 6)` | Counts restricted to synapses selecting each class |
| `selected_class_sums` | `(10, 6)` | Selected-class probability sums within those restricted groups |
| `selected_class_squares` | `(10, 6)` | Corresponding sums of squared probabilities |
| `classes` | `(6,)` | Class names |
| `provenance` | scalar string | JSON: format version 2, model-plan SHA-256, family, fold |

Embedding-count index 0 means **one embedding**, and index 9 means ten. Counts
are the actual number used by that model: up to ten for CAVE n10, one for
single_pre_post. The `selected_class_*` arrays drive the current six-class
plots; the older all-synapse `sums[:, 1:]` remain available but are not used for
those panels.

| Fold | Included CAVE n10 synapses |
|---:|---:|
| 0 | 8,006,943 |
| 1 | 8,009,607 |
| 2 | 8,014,751 |
| 3 | 8,001,919 |
| 4 | 8,014,413 |

### Read the mean probability for a predicted class

```python
with np.load(cache / 'unlabeled_fold0.npz', allow_pickle=False) as z:
    j = z['classes'].tolist().index('BasketFam')
    n = z['selected_class_n'][:, j]
    total = z['selected_class_sums'][:, j]
    mean = np.divide(total, n, out=np.full(10, np.nan), where=n > 0)
# mean[k-1] averages P(BasketFam) only among BasketFam predictions using k nodes.
```

The six-class figure first computes those means separately for each fold, then
averages with **equal fold weights**. Shading represents nested 50%, 80%, and
95% Student-t confidence intervals across available fold means. The separate
single-panel graph retains one line per fold, with point-level normal intervals.
Neither interval method measures accuracy; the data are unlabeled. Shared roots,
fragments, and overlapping fold prediction cohorts introduce correlations.

Histogram composites pool fold records rather than averaging probabilities
across folds. Density is `count / (total count × bin width)`, so each displayed
histogram has area one; its height may exceed one.

## Access individual deployment rows

Use DuckDB in read-only mode and select only needed columns. The class index is
zero-based, while DuckDB list indexing is one-based.

```python
from pathlib import Path
import duckdb

source = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/'
              'incoming_presynaptic_fragments_k10_collina_v2/cell_types')
con = duckdb.connect(str(source / 'predictions.duckdb'), read_only=True)
rows = con.execute('''
    SELECT synapse_id, cell_root_id, partner_root_id, anchor_node_id,
           cave_n10_six_fold0_n_embeddings_used AS embeddings_used,
           cave_n10_six_fold0_level2_class_index AS selected_class,
           cave_n10_six_fold0_level2_probabilities AS probabilities,
           cave_n10_six_fold0_level2_probabilities[
               cave_n10_six_fold0_level2_class_index + 1
           ] AS selected_probability
    FROM cave_n10_six_fold0
    LIMIT 10
''').df()
con.close()
```

Identity fields:

- `synapse_id`: synapse identity; never collapse rows merely because pre fragments match.
- `cell_root_id`: postsynaptic target root.
- `partner_root_id`: presynaptic partner root.
- `anchor_node_id`: presynaptic skeleton anchor.
- `fragment_node_ids`: selected presynaptic skeleton nodes.
- `n_embeddings_used`, `n_embeddings_available`, `missing_embedding_count`, `status`: source fragment coverage.
- `presynaptic_fold{f}_class`, `presynaptic_fold{f}_is_axon`: per-fold subcompartment decisions/filter.
- `postsynaptic_fold{f}_class`, `postsynaptic_fold{f}_logits`: post point subcompartment predictions, retained without an extra post filter.

Cell-type columns use `{family}_{classes}_fold{fold}` prefixes, for example
`cave_n10_six_fold0`. Each has `_included`, `_n_embeddings_used`, and
`_level{level}_logits`, `_level{level}_probabilities`, `_level{level}_class_index`.
For the six-class hierarchy, level 2 is the six-class leaf level. The DuckDB
`predictions` view includes all source synapses; the 20 model/fold views filter
to included rows. Excluded model/fold scores are null.

## Rebuild and render

From the repository root:

```bash
# Build both cache types for CAVE n10: five concurrent CPU tasks.
sbatch analysis/db_deployment/presynaptic/20261009/build_cache.sh

# Rebuild only unlabeled summaries; no model inference required.
sbatch analysis/db_deployment/presynaptic/20261009/build_cache.sh --stage unlabeled

# Optional alternative family, before changing notebook FAMILY.
sbatch analysis/db_deployment/presynaptic/20261009/build_cache.sh --family single_pre_post
```

Jobs use `mit_amf_advanced_cpu`. Matching caches are reused; writes are atomic.
Unlabeled cache construction streams only needed columns from prediction Parquet
files, in batches. Test cache construction runs the saved model on held-out data.

After jobs finish, Run All in the notebooks. `render_notebooks.py` also executes
both notebooks and saves their figure outputs; use an environment with this
repository's dependencies and its Python on `PATH` for the Jupyter kernel.

Repository files:

| File | Purpose |
|---|---|
| `test_analysis.ipynb` | Held-out probability, accuracy, and histogram confusion-matrix figures |
| `unlabeled_analysis.ipynb` | Deployment probability/embedding-count figures |
| `cache_analysis.py` | Cache builders, paths, class/provenance checks |
| `plot_helpers.py` | Shared plotting and confidence-band calculations |
| `build_cache.sh` | Parallel CPU cache-building array |
| `render_notebooks.py` | Execute and save both notebooks |
