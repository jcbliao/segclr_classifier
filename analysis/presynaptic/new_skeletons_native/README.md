# Resampled TEASAR analysis: 2,209 cells

- `skeleton_stats.ipynb`: summary, three-component Gaussian mixtures in log
  and linear space, cell-type distributions, and upstream branch/site plots.
- `presynaptic_axon_database_figures.ipynb`: nearest-native-node distance
  distribution, per-cell means, and rejection fractions by cell type.
- Previous CAVE/SegCLR diagnostics:
  `../cave_skeletons/presynaptic_axon_database_figures.ipynb`.

The cohort is exactly the 2,209 NPZ records in `presynaptic_axons/k10/cells`.
Resampled geometry follows `segclr_registered_teasar_111nm_20260911/routes.json`:
2,179 registered sources and 30 reused testing sources. Resampling splits edges
to at most approximately 111 nm; it preserves original short edges and geometry.

The new database is
`/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/native`.
It reuses saved presynaptic coordinates and soma-cut centers from k10, applies
the 5 µm soma exclusion, then maps sites to the nearest surviving resampled
node. No embeddings or K-node windows are required.

Every analysis selects connected components with **strictly more than five
unique presynaptic IDs mapped within 2 µm**. All their edges are retained.
Upstream analyses additionally exclude components that are not trees with
exactly one soma-cut boundary root. Distance diagnostics include rejected
point rows when their nearest node is on a qualifying component; those rows
do not contribute to component qualification.

Regenerate from the repository root using the ossify environment:

```bash
python data/build_native_presynaptic_stats.py --workers 8
python analysis/presynaptic/new_skeletons_native/run_notebooks.py
```

Use a batch allocation for the initial build. Notebook reruns reuse persistent
edge arrays, summary/quantile/mixture caches, and upstream histograms.
Source file signatures invalidate caches after data changes.
`verification.json` records the completed notebook population checks.

## Training fragment lengths

Registered native training uses
`{1: 257, 2: 129, 4: 65, 8: 33, 16: 17, 32: 9}` (scale: nodes).
The rule is `K = 256 / scale + 1`: each finer scale doubles the fragment edge
count, preserving approximate cable length while including the center node.
The queued experiment uses scale 32 / K=9. Fragment sizes are fixed by node
count, **not a length eligibility cutoff**. Longer fragments remain eligible.
Soma exclusion, the 2 µm site matching
cutoff, and the requirement that a component contain K nodes remain unchanged.

The nearest eligible full-resolution node is now resolved **before** subsampling
(after the existing soma exclusion). Each fragment locally replaces the closest
degree-2 endpoint of its containing coarse edge with that exact original node.
The two incident edge lengths are adjusted along the original cable path. If
both endpoints are branch/end anchors, its center is inserted on the edge for
that window only. Branch/end anchors are never moved. The K nearest nodes are
then selected by cable distance, including the center as the first of K nodes.
Exact original IDs select the features; no embeddings or coordinates are
interpolated. Window-local edges and LPE are stored explicitly, and the loader
uses those edges instead of the dense geometry's induced adjacency.
Training uses `resampled_teasar_2209/training_local_center` to keep previous
nearest-retained-node caches separate.

Cable length is the sum of the fragment's undirected edge weights, counting
each edge once. Contracted edges retain original cable lengths. This differs
from the maximum distance to the center (`new_radius_nm`) and from Euclidean
endpoint distances. Training outputs now also store `new_cable_length_nm`.

`scripts/plot_presynaptic_fragment_lengths.py` measures the old K=7 and new
configured K distributions directly from the native geometry, independently of
embedding completion. It uses the production window selector and deduplicates
by root-scoped sorted node membership, matching the training loader.
Full-cohort PNG/PDF histograms, numeric arrays, and summary statistics go to
`fragment_lengths_local_center/k9/`. Regenerate with a 32-task array of
`scripts/sbatch/plot_presynaptic_fragment_lengths.sh`, followed by the same
batch script with `--plot` after all array tasks succeed.

Current K=9 measurements on the existing threshold-resampled native skeletons:
2,303,295 unique fragments across 2,209 cells, mean 21.035 µm, median 20.929 µm,
5th–95th percentiles 18.620–23.762 µm, and 1.184% above 25 µm (retained).
There are 2,370,538 valid site rows. These are not measurements of the separate
`teasar_target` geometry, which has been registered without inference.

Previous window-local-center measurements across all 2,209 cells: K=10 has
2,302,914 unique fragments, mean 23.665 µm, median 23.559 µm, modal 1 µm bin
23–24 µm, and 5th–95th percentiles 21.041–26.591 µm. 20.47% exceed 25 µm
and remain included. There are 2,370,519 valid site rows. All valid windows
passed the exact full-resolution-center membership assertion. Synthetic checks
cover every scale, degree-2 replacement, immutable-anchor insertion, exact
cable lengths, and long-window retention; a real-cell loader check verifies
center embedding identity, fixed K, and connected window-local graphs.

Historical nearest-retained-node measurements in `fragment_lengths/` (before
preserving full-resolution centers; superseded for training):

| Scale-32 nodes | Fragments | Mean cable (µm) | Median (µm) | 5th–95th percentile (µm) |
|---|---:|---:|---:|---:|
| 7 (previous) | 1,719,192 | 15.80 | 15.69 | 13.85–18.09 |
| 10 (previous) | 1,647,777 | 23.64 | 23.51 | 21.08–26.60 |

That historical histogram's modal 1 µm bin is 23–24 µm. 20.03% of fragments
exceed 25 µm and remain included; the maximum is 35.16 µm. A smaller unique
fragment count at K=10 reflects membership deduplication and small-component
eligibility, not cable-length filtering. Valid site rows are 2,336,213 versus
2,336,250 at K=7.

Obsolete 38-cell notebook outputs, component plots, and the old plotting script
were removed from this directory. Shared resampled source skeletons are retained:
30 of them are also part of the requested 2,209-cell cohort.
