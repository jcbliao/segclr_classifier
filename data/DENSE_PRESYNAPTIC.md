# Dense TEASAR presynaptic datasets

Input is the 466-cell testing cohort's **upsampled** named skeleton
`teasar_testing_111nm_20260910`, with directly inferred 64-dimensional embeddings
from `resnet_860b_reshuffled__20260603_150412/checkpoint_e0_s95000`.

| Skeleton node fraction | Window embeddings, including the nearest node |
|---|---:|
| Full | 209 |
| 1/2 | 105 |
| 1/4 | 53 |
| 1/8 | 27 |
| 1/16 | 13 |
| 1/32 | 7 |

`data/build_dense_presynaptic.py` has five resumable training stages:

1. `--prepare`: freeze cohort, soma positions, proofread-axon eligibility and
   splits. Existing cells retain their previous train/test assignment; new
   cells use the same seed-0 stratified splitting function. All scales share
   this manifest. The previous v1718 proofreading-status snapshot is reused.
   Soma coordinates use the shared cell table first, then the saved cohort's
   CAVE soma coordinates in nanometers when missing from that table.
2. `--topology`: for **every** source cell, remove nodes at distance <=5 µm
   from the CAVE node closest to the cached soma, then construct nested
   subsamples. Keep all branch points, endpoints and isolated nodes. Each
   target is ceil(post-cut node count / factor); topology anchors take
   precedence if that target is impossible. Missing soma locations are
   recorded: those cells get uncut topology but cannot enter training.
3. `--sites`: fetch outgoing/presynaptic synapses for eligible proofread axons
   using the existing v1718 synapse-query and normalization code. Results are
   cached per cell, including valid empty results.
4. `--windows`: require complete inference for the cell, locate each
   presynaptic site on the nearest retained node within 2 µm, and collect the
   K nearest nodes by cable distance, **including the center**. Components
   smaller than K are rejected. Generate the same window-local LPE as before.
5. `--finalize`: verify coverage of every eligible cell and publish one
   training manifest/metadata pair per scale. Incomplete builds are rejected
   by training rather than silently shrinking the cohort.

Subsampling removes only degree-2 vertices. Each replacement edge stores the
sum of its constituent edge lengths, so geodesic distances between retained
nodes and total cable length are preserved even along curved paths. Retained
coordinates are never interpolated. Edge spacing roughly doubles as the node
budget halves; branch-point constraints and original spacing prevent an exact
uniform doubling everywhere. Cyclic inputs are rejected rather than pruned.

Geometry files retain **original dense skeleton node IDs**. Features are
selected by exact ID from the inference NPZ files; no spatial embedding match,
averaging, or re-inference is performed. All six scales reference the same
embedding file. Duplicate windows are removed by the existing loader's
root-scoped exact-membership rule.

Default output:
`/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/dense_teasar_multiscale_v2`

## Commands

All computation runs through SLURM. Preparation and topology can run before
inference completes:

```bash
MODE=prepare sbatch scripts/sbatch/build_dense_presynaptic.sh
# After preparation succeeds:
MODE=topology NUM_TASKS=32 sbatch --array=0-31%16 scripts/sbatch/build_dense_presynaptic.sh
```

After inference is complete, submit the remaining pipeline (existing topology
and sites are reused). Alternatively supply the outstanding inference job IDs:

```bash
bash scripts/submit_dense_presynaptic.sh
# Or schedule it behind specific successful inference jobs:
INFERENCE_JOB_IDS=12345:12346 bash scripts/submit_dense_presynaptic.sh
```

Retain the inference output directory: the dataset references its immutable
per-cell embedding files. The shared database also stores these vectors in
`named_node_embeddings`; the dataset uses the already validated files to avoid
repeated database scans. Outputs here are training caches, not database writes.

Train using the existing training entry point from a GPU batch job, e.g.:

```bash
python -u scripts/train_gnn.py --dataset presynaptic_dense \
  --presynaptic-database /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/dense_teasar_multiscale_v2/scale1/k209 \
  --num-embeddings 209 --architecture mean --class-balance sample
```

Choose the paired scale/K directory for each experiment. The loader supports
the embedding-only mean/linear/pointwise MLP models and both transformer
implementations. Result paths include the scale and K, avoiding collisions.
The existing attention-budget batching automatically reduces batch window
counts for the larger graphs. Training is not submitted by the build script.

## Native skeleton analysis without embeddings

`analysis/presynaptic/new_skeletons_native/skeleton_stats.ipynb` reuses the
original notebook's statistics and plotting implementation for the full
upsampled skeletons after soma exclusion. It includes edge-length fits,
cell-type panels, and upstream branch and presynaptic-site counts.

After topology and sites finish, build its geometry-only cache:

```bash
MODE=native NUM_TASKS=32 sbatch --array=0-31%8 scripts/sbatch/build_dense_presynaptic.sh
# After all native tasks succeed:
MODE=native-finalize sbatch scripts/sbatch/build_dense_presynaptic.sh
```

The cache lives under `dense_teasar_multiscale_v2/native`. It includes the
same proofread-axon eligibility and soma exclusion as training. Presynaptic
sites map directly to retained geometry nodes within 2 µm, without requiring
inference, a K-node window, or any embedding files. This is scale 1: no
subsampling is applied. All six topology scales remain available for the
subsequent training dataset build.
