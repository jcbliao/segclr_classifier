# Dendrite-filtered presynaptic means

**Current requested analysis:** [Casey L2 2D/3D comparison](casey_l2/README.md), balanced to 2,000 windows per each of six L2 classes. The 13-class plots in this parent directory are the initial coarse-type visualization, retained for reference. The plotting script now defaults to zero-based hierarchy level 2.

Source: native scale16/k17, confidence 0.7, fold0 presynaptic windows with Casey coarse cell types and thalamocortical cells. The source manifest contains 1,773 labeled cells. Classifier: the geodesic 20 µm soma-exclusion inverse-square-root ResNet checkpoint, with its saved training normalization.

Every observed SegCLR embedding receives a compartment prediction on the GPU. A window is removed if **strictly more than half** of its observed embeddings predict dendrite. Other compartment predictions and ties are retained. Invalid source windows remain excluded. Voting happens before deduplication by the full sorted node set within each cell. This is a window filter; it does not remove whole cells or connected skeleton components.

Means use raw SegCLR features of the retained window's observed nodes. All means and masks are saved on scratch. UMAP samples uniformly without replacement within each cell-type class, with **exactly equal class counts**: min(2,000, smallest retained class). The UMAP is unsupervised, with no feature standardization: Euclidean metric, 15 neighbors, min_dist 0.1, 500 epochs, seed 0. Both original manifest splits are included; this is a visualization, not an evaluation of cell-type generalization. The saved checkpoint was selected using compartment test macro F1.

## Outputs

- `mean_embedding_umap.png` and `.pdf`: class-colored UMAP.
- `balanced_mean_embeddings.npz`: sampled input embeddings with root IDs, source window rows, cell types, and splits.
- `umap_coordinates.npz` and `.csv`: coordinates with matching identifiers.
- `reducer.joblib` and `settings.json`: fitted UMAP and provenance.
- `cell_audit.csv`, `cell_type_audit.csv`, and `filter_summary.json`: complete filtering counts, independent of UMAP subsampling.

Full mask/mean cache:
`/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/compartment_filtered/scale16/k17/conf0.7/fold0/geodesic_exclusion20um`.
Each cell NPZ contains site-aligned `keep` and `votes`, node predictions, unique retained `window_rows`, `mean_embeddings`, and `cell_mean` (equal-weight mean of retained window means, not a mean of unique cell nodes).

## Reproduce

```bash
filter_job=$(sbatch --parsable scripts/sbatch/filter_presynaptic_compartments.sh)
sbatch --dependency=afterok:$filter_job scripts/sbatch/plot_filtered_presynaptic_umap.sh
sbatch --dependency=afterok:$filter_job scripts/sbatch/check_presynaptic_compartment_filter.sh
```

Training can reuse the same masks with:

```text
--presynaptic-compartment-filter /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/compartment_filtered/scale16/k17/conf0.7/fold0/geodesic_exclusion20um
```

The loader verifies the source database, skeleton variant, and site IDs. It filters before the existing postsynaptic eligibility and membership deduplication. If no results directory is supplied, filtered runs use a separate results subtree. Existing trained checkpoints are unchanged; no retraining was requested or submitted.

Initial jobs: GPU filter `24691643`, CPU UMAP `24691677`, loader verification `24691719`.
