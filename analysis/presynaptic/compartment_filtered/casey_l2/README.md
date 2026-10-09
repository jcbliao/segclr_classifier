# Presynaptic window means colored by Casey L2

Labels come from **zero-based hierarchy level 2** of the source database's Casey `hierarchy_tree`, matching the classification target after dropping the two finest levels. The six represented classes are BasketFam, BipFam, MartFam, NglFam, pyramidal, and thalamocortical. All eight cortical excitatory coarse types map to pyramidal.

Sampling is uniform without replacement over retained windows within each **L2 class**, exactly 2,000 per class (12,000 total). Both UMAP fits use the identical 64-dimensional raw means and sample ordering. Source coarse labels are preserved as `source_cell_type`; root IDs are strings so interactive hovering preserves all digits.

The dendrite-majority filter is unchanged: 39,780 of 2,048,716 unique source windows removed. Source data and trained models are preserved.

## Artifacts

- `mean_embedding_umap_2d_3d.html`: interactive 2D/rotatable 3D with class toggles and source-type hover details. Plotly loads from the same CDN used by the existing figure exporter; opening the interactive page needs internet access.
- `mean_embedding_umap_2d.png`, `.pdf`, `mean_embedding_umap_3d.png`, `.pdf`: static projections using the same class palette.
- `umap_coordinates.npz` (2D), `umap_coordinates_3d.npz`, and `umap_coordinates_2d.csv` / `_3d.csv`: aligned coordinates, means, and metadata.
- `dimension_comparison.csv`: neighborhood preservation and same-class neighbor agreement.
- `dimension_comparison_settings.json`: palette, label mapping, fit settings, and comparison metrics.

UMAP parameters: Euclidean metric, 15 neighbors, min_dist 0.1, 500 epochs, seed 0, no standardization, no supervision. Trustworthiness uses the same random 100 windows per L2 class in both projections (600 total), with 15 neighbors. Same-class neighbor agreement uses the 15 nearest neighbors of all 12,000 projected points. These are descriptive comparisons, not held-out cell-type classifier scores; different windows from the same neuron can be neighbors.

## Reproduce

```bash
job=$(sbatch --parsable scripts/sbatch/plot_filtered_presynaptic_umap.sh --hierarchy-level 2 --output analysis/presynaptic/compartment_filtered/casey_l2)
sbatch --dependency=afterok:$job scripts/sbatch/compare_filtered_umap_dimensions.sh --output analysis/presynaptic/compartment_filtered/casey_l2
```

Jobs: L2 sampling / 2D fit `24692547`, 3D fit / comparison / figure export `24692648`.
