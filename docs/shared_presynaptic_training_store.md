# Shared presynaptic training inputs

The newest-SegCLR prepared window and feature stores are published at:

```text
/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/training_stores/latest100k_morpho6_conf0_v1
```

Check `complete.json` for a successful verification receipt before using a
publication. This directory is readable by `orcd_rg_fstor012_pi_sdorkenw`.
Jacob's private home and scratch directories are not needed to read the bundle.

It includes CAVE K10 pre/post and native single pre/post inputs for 2,029 cells,
five folds, six morphological classes, with no label-confidence cutoff. Both
use the newest pinned SegCLR checkpoint, `checkpoint_e0_s100000`. The original
window membership, center selection, labels and fold assignments are retained.

The bundle contains:

- Per-cell window arrays and newest embeddings, with fold views.
- Materialized native geometry files and rewritten embedded references.
- One shared per-cell postsynaptic feature store under `postsynaptic_features/`.
  CAVE and native caches retain separate synapse IDs, eligibility flags and row
  maps, with relative references to those features. Source site/feature files
  are retained for provenance and rebuilding.
- Shared fold manifests and ten training recipes under `configs/`.
- A `train.sh` entry point, dependency hashes, and a verification receipt.

Native features cover the selected centers consumed by the single-input model.
They do not cover every unused vertex of the full native geometry. Alternative
models that need embeddings at additional native vertices must generate or
retrieve those features separately.

## Train from your own checkout

Install the repository's training dependencies in your own environment, then
run on an allocated GPU from your `segclr_classifier` checkout:

```bash
PYTHON=/path/to/your/environment/bin/python \
  /orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/training_stores/latest100k_morpho6_conf0_v1/train.sh \
  cave_n10 0
```

Use `single_pre_post` for the native model, and fold numbers 0 through 4.
`GNN_CLASSIFIER_REPO` can specify the checkout explicitly. Configurations write
results under `results/shared_latest100k` in that checkout. Runtime packed
arrays use the job's temporary directory by default; set
`SEGCLR_PACKING_CACHE` to another writable location when needed. Successful
training removes its temporary packing caches. The shared prepared window and
feature stores are retained.

These prepared stores can train without live `segclr_db` access. Model registry
and prediction-import status are separate from this publication.

Shared postsynaptic features require a checkout containing
`data/postsynaptic_cache.py`; that loader also supports the older inline caches.
The owner can migrate an existing bundle on a CPU allocation with
`python scripts/share_postsynaptic_features.py /path/to/bundle`. Migration
checks every reconstructed feature array byte for byte and preserves eligibility
and synapse ordering. New publications perform this step automatically.

## Publish or verify

The owner can run `scripts/publish_presynaptic_training_store.py build` on a CPU
allocation to publish, or `verify` to check an existing bundle. Publication
copies inputs, materializes dependencies, rewrites paths, and grants the lab
group access only to files in the new bundle. Source data and private directory
permissions are unchanged.
