# Paired axon-only versus axon + dendrite training

Confirmed experiment: five CAVE folds × two pointwise MLP training conditions.
Both conditions classify exactly the same held-out presynaptic axon windows.
This tests the effect of adding dendritic training examples; it excludes soma
windows and should not be described as training on entire cells including somata.

## Inputs and models

- Original confidence ≥0.7 Morpho coarse/L2 neuron cohort and five outer cell splits.
- Six labels: BasketFam, BipFam, MartFam, NglFam, pyramidal, thalamocortical.
- Nearest 10 connected CAVE skeleton embedding nodes; mean pooling occurs
  **after** the nonlinear 64→128→128 pointwise MLP (GELU), not before it.
- A 64D contextual input is concatenated after pooling: original postsynaptic
  point embedding on presynaptic axon windows; raw node-embedding mean on
  dendritic windows. Both models therefore have the same architecture and width.
- Existing four-layer, width-128 ResNet trunk and LCPN hierarchy heads.
- Axons: original eligible presynaptic CAVE n10 windows, with the source loader's
  per-root node-set deduplication, plus fold-specific hard axon filtering.
- Dendrites: unique node-set neighborhoods from the existing whole-cell n10
  cache after its 5µm Euclidean nucleus/soma exclusion, plus hard dendrite filtering.
- Filters classify the **raw mean of the window embeddings**, using the existing
  geodesic 10µm soma-exclusion subcompartment model for the corresponding fold.
  They do not vote over node labels. Filter checkpoint hashes and class orders
  are pinned in the plan. Both arms share exactly the same axon filter.
- Thirteen thalamocortical roots lack cached nuclei/whole-cell neighborhoods.
  Their eligible axon windows remain in both arms; they provide no dendrite
  examples. No nucleus location is invented or soma exclusion bypassed.

## Splits and comparison controls

Outer test roots are unchanged. Within the original outer-training roots,
approximately 10% are selected as inner validation cells, stratified by the
six classes and restricted to roots with retained axon windows. All remaining
training roots are shared across both conditions. Training, validation, and test
roots are disjoint. Dendritic windows are added only from training roots.

The best checkpoint maximizes **inner-validation axon-window macro F1**.
Outer-test axon windows are evaluated only after training and reloading that
checkpoint. Both arms use the same validation/test index files, not separately
sampled evaluation subsets. Cell metrics majority-vote the axon predictions.

Each epoch draws exactly as many windows as the axon-only training set contains;
both arms use identical batch sizes and optimizer-step counts. Balanced sampling
uses the axon-only reference class distribution (expected class shares
proportional to sqrt(reference class window count), corresponding to 1/sqrt(count)
window weights). Within each class, sample uniformly from the condition's
available windows. Thus both conditions draw the same class counts at each
seed/epoch, even though the mixed pool contains additional dendritic examples.
Samples are grouped for efficient reads without changing the sampled multiset.

Both arms initialize from the same random seed. Train for 30 epochs with AdamW,
learning rate 1e-4, weight decay 1e-5, batch size 4096, BF16 autocast, unweighted
hierarchical loss, and gradient norm clipping at 1. Resume restores optimizer,
model, and RNG state; historical experiments/checkpoints remain untouched.

Subcompartment models are frozen auxiliary models with their own training splits;
this experiment does not retrain them on the new inner cell-type splits. Their
fold numbering should not be interpreted as proving cell-disjoint auxiliary
training relative to these cell-type evaluation sets.

## Storage and entry points

Prepared inputs and shared split indices:

`/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/axon_vs_whole_cell_casey6_20261009/`

- `plan.json`: frozen source signatures, model hashes, labels, targets, settings.
- `cells/<root>/`: memory-mapped `.npy` node bags, masks, contextual vectors,
  per-fold subcompartment calls, synapse IDs/center indices, completion counts.
- `indices/`: per-fold train-axon-only, train-mixed, shared-validation,
  shared-test row indices, and cell split manifests.
- `complete.json`: successful preparation and paired-index audit.

Training outputs:

`results/presynaptic/axon_vs_whole_cell_morpho6/fold{0..4}/{axon_only,axon_dendrite}/`

Each run saves best/last checkpoints, epoch training/validation metrics and timing,
final outer-test window/cell metrics, and `test_predictions.npz` with identities,
true labels, selected classes, joint six-class probabilities, and conditional
per-level logits. Nothing is written to the shared segclr_db.

Implementation: `scripts/paired_axon_experiment.py` (initialize/prepare/finalize),
`scripts/train_paired_axon_experiment.py` (training), and corresponding wrappers
in `scripts/sbatch/`. Tests: `scripts/test_paired_axon_experiment.py`.

Preparation uses parallel CPU jobs under `mit_amf_advanced_cpu`. Training requests
16 CPUs, 15 data-loader workers, one L40S GPU under `mit_amf_advanced_gpu`, and
at most four simultaneous GPU tasks. GPU training must wait for the earlier
balanced-sampling array to finish so the combined workload stays within four GPUs.
