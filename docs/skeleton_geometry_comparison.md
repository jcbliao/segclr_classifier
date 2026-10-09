Compare full CAVE and TEASAR skeletons for cells in
`data/v1718_extended_axon_neurons_casey_comparison.csv` with a Casey identity.
This CSV joins the v1718 extended-axon list to Casey by root or stable nucleus
identity. There is no additional label, confidence, split, or cell-type filter.
Both skeletons use the v1718 root: Casey's newer root is label metadata only.

Run from the repository:

```bash
bash scripts/submit_skeleton_geometry.sh
```

The pipeline tests the distance implementation, exports CAVE geometry once
from the v1718 Lance store, freezes available pairs, runs 16 array shards
(up to 8 simultaneously, 4 processes each), then writes `metrics.csv` and
`summary.json`. No live CAVE calls, tokens, embeddings, or GPUs are needed.
Jobs use the existing `mit_amf_advanced_cpu` account and QoS on `mit_normal`.
TEASAR geometry comes from the original skeleton files selected by the existing
registered-cohort routes. Missing matches or skeletons are listed in the cohort
audit rather than silently dropped. The default output is
`/orcd/scratch/orcd/013/jcbliao/skeleton_geometry_comparison`.

```bash
# Change parallelism or integration accuracy; use a separate output for variants.
NUM_TASKS=32 MAX_CONCURRENT=8 bash scripts/submit_skeleton_geometry.sh \
  --output /orcd/scratch/orcd/013/jcbliao/skeleton_geometry_fine --spacing-nm 100

# Resume without exporting CAVE again (unchanged successful cells are skipped).
NUM_TASKS=16 sbatch --array=0-15%8 scripts/sbatch/compare_skeleton_geometry.sh run --retry-errors
sbatch --dependency=afterany:ARRAY_JOB_ID scripts/sbatch/compare_skeleton_geometry.sh summarize
```

Each cell has an atomic JSON checkpoint under `cells/`, including input file
timestamps and parameters. Changed inputs or parameters trigger recomputation.
Avoid overlapping runs against the same output directory. A fresh full pipeline
exports CAVE again, which changes the input timestamps and recomputes results;
use the resume command for interrupted comparisons.

`precision_*` is the fraction of TEASAR cable within a tolerance of CAVE;
`recall_*` reverses that direction. `f1_*` combines them at 0.25, 0.5, 1, and
2 µm. These measure agreement, without asserting that CAVE is ground truth.
The symmetric mean is the average of the two directional, cable-weighted means.
`*_p95_um` is a cable-weighted distance percentile, and `*_sampled_max_um` is
the largest distance at a sampled source midpoint, not a true Hausdorff distance.

Target distances are **exact point-to-segment distances**, accelerated by a
KD tree with a proven stopping bound. Original target edges are subdivided into
at most 1 µm segments without changing geometry. Source cable is integrated
with midpoint subdivisions of at most 250 nm, each weighted by its own cable
length. The mean distance has a conservative quadrature error bound of 125 nm
at the default spacing. Overlap includes lower/upper bounds accounting for
variation within every source subdivision. Reducing spacing tightens these
bounds. Raw vertex density therefore does not determine the weighting.

Full skeletons are compared without translation, registration, soma cuts, or
component pruning. Verify the scope before comparing against soma-cut training
graphs. Length ratios, components, isolated nodes, endpoints, branch nodes, and
cycle rank are diagnostics, not matched-topology scores. Branch-node counts can
depend on how a branching junction is represented. DIADEM and matched-location
geodesic distances are not implemented by this runner.

The summary reports medians across cells and explicit missing/error/stale lists;
it exits unsuccessfully if the frozen cohort is incomplete. Per-cell metrics
retain the original and Casey labels and match source for stratified analysis.

The initial run on 2026-10-01 completed all 2,069 paired cells with no errors,
missing results, or stale checkpoints. Of the 2,103 input cells, 34 had no Casey
identity match; every matched cell had both skeletons. Seven analytic and
brute-force tests passed. Preparation took 2m59s; comparison shards took roughly
12–16s for the first wave. Cell medians were 0.190 µm symmetric mean distance,
0.937 overlap F1 at 0.5 µm, 0.987 at 1 µm, and a TEASAR/CAVE cable-length ratio
of 1.087. These describe agreement of the full skeletons, not reconstruction
accuracy against ground truth.
