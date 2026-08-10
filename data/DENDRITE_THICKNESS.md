# Dendrite thickness: what it is and how to run it

See CLAUDE.md's "Dendrite thickness" section for how this fits into the project and for the
coverage caveat that governs how to read any result from it; this file is the practical how-to
for the ingestion pipeline.

## What it computes

For every eligible skeleton vertex of a cell (dendrite compartment, not a branch point, a
well-defined local tangent), a **spine-corrected shaft radius**: rays are cast from the vertex
within the plane perpendicular to the local skeleton tangent, and the median first-hit distance
against a local mesh patch (with a few passes of recentring) approaches the true shaft radius.
Confining rays to that plane is what rejects spines — a spine is near point-like along its own
axis, so a ray only meets one if its mouth happens to cut exactly that plane. The `radius`
already stored on CAVE skeletons (`Skeleton.radii`) has no such correction and reports "shaft
plus spine heads," which inflates cross-section and flattens apparent taper.

Ported from a private repo, `isebenius/E-I`'s `src/dendrite_thickness.py` (fetched via `gh api`,
per explicit user direction, 2026-08-07). The estimator math (`data/dendrite_thickness.py`,
`skeleton_neighbors` through `estimate_dendrite_radius`) is unchanged from the original, with
one deliberate substitution: branch-point detection uses `degree > 2` computed directly from
the skeleton's own edges, instead of the original's `skeleton.branch_points` (a
`meshparty.skeleton.Skeleton`-specific property) — so the estimator only ever needs
`.vertices`/`.edges` and works against this project's own `Skeleton` dataclass without building
a real meshparty object per cell.

## Alignment

- **Skeleton-to-skeleton**: already exact, no work needed. The cached whole-cell `Data`
  (`data/graph_cache/*.pt`) carries `orig_node_ids`, the index of each covered node back into
  its source `Skeleton`'s vertex array — `data.pos[i] == skeleton.coords[data.orig_node_ids[i]]`
  by construction, since both came from the same `Skeleton` object.
- **Mesh-to-skeleton**: not assumed, checked empirically per cell via `miss_fraction` (fraction
  of rays that hit nothing) and radius sanity — logged as a warning when a cell has eligible
  vertices but zero measured ones, which would flag a coordinate-frame problem if one ever
  showed up. None has, across the full corpus.

## How to run

Everything goes through the SLURM array wrapper (`scripts/sbatch/build_dendrite_thickness.sh`,
`mit_preemptable`, `#SBATCH --requeue`) — never run `data/build_dendrite_thickness.py` directly
(see CLAUDE.md's hard constraints).

```bash
# Full corpus, default 8-way weight-balanced shard, 4 concurrent (0-7%4)
sbatch scripts/sbatch/build_dendrite_thickness.sh

# Cheap validation pass -- LIMIT truncates each array task's per-shard todo list
LIMIT=2 sbatch scripts/sbatch/build_dendrite_thickness.sh

# Backfill just the handful of cells still missing a cache file -- one task,
# no sharding, since the script's resumability check (below) skips everything
# already done. Override --array on the sbatch command line, not via an env
# var -- #SBATCH directives are fixed at script-parse time.
sbatch --array=0-0 scripts/sbatch/build_dendrite_thickness.sh

# Retune concurrency (per-array-task worker threads)
WORKERS=2 sbatch scripts/sbatch/build_dendrite_thickness.sh
```

**Resumable by construction, at two levels:**
- Per cell: `data/build_dendrite_thickness.py::main` only builds a `todo` list of cells whose
  `data/dendrite_thickness_cache/{root_id}.npz` doesn't already exist, and only calls
  `np.savez` after a cell fully completes — so a killed/preempted/requeued task just re-scans
  and continues, no partial files.
- Per shard: shards are a fixed, deterministic partition of *all* root_ids
  (`balanced_shards`, weighted by `manifest.json`'s `n_nodes_covered` as a proxy for expected
  ray-casting work), so shard membership never shifts across separate runs of the array —
  task `N` always means the same cells, whether or not they're already done.

## Output

`data/dendrite_thickness_cache/{root_id}.npz` — a single array, `radius_nm`, shape
`(n_skeleton_vertices,)`, float32, indexed the same way `data/geodesic_window.py` already
indexes `pos`/`compartments` for that cell (i.e. via `orig_node_ids`, see "Alignment" above).
`NaN` where unmeasured: non-dendrite compartment, branch point, or a mesh-hole/miss.

## Status

**2192/2192 cells complete** — the full corpus (job `19889023`, 8-shard array, plus a
single-task backfill for `864691135014099446`, which had hit a transient `502 Bad Gateway` from
CAVE's mesh manifest endpoint rather than a real failure). Verified by
`scripts/check_thickness_features.py`: zero cells missing a cache file.

**Now consumed by the model**, opt-in via `scripts/train_gnn.py --gt-use-thickness` (off by
default, since it requires this cache). That one flag turns on both the dataset side
(`WindowedGraphDatasetLCPN(..., use_thickness=True)`, which joins the cache through
`orig_node_ids` and normalizes/NaN-masks once per cell) and the model side
(`ModelConfig.gt_use_thickness`, which concatenates it onto the node features alongside
`rel_pos`). Only `--architecture graph_transformer` reads it. Runs are tagged `_thick`.

The feature is two channels — normalized radius and a *measured* flag — because NaN is the
normal value here, not an error. `scripts/check_thickness_features.py` validates the join
against the raw `.npz` and reports coverage.

**Parked by explicit user direction — off by default, and not recommended.** The reason is not
that the measurement failed: of the 3.63M nodes where a shaft radius is defined, 100.0% were
measured (ray-cast failure rate 0.03%). It is that only 30.8% of nodes are eligible at all,
because **68.7% are axon**, where the quantity is undefined. So the measured flag is
effectively a dendrite-vs-axon indicator, and since axon share runs from ~41% (L5ET) to ~83%
(PV), that indicator separates E/I by itself. Any gain from this feature would need a
mask-only control run to attribute. `scripts/check_thickness_coverage_breakdown.py` regenerates
the decomposition; CLAUDE.md's dendrite-thickness section has the full numbers.

## Validation against known geometry

`data/dendrite_thickness.py::validate_on_synthetic_cylinder` builds a cylinder mesh with a known
shaft radius and spine-like protrusions (via `trimesh.boolean.union`) and a skeleton
deliberately offset from and tilted relative to the true axis, then checks three things at
once: the perpendicular-ring estimate recovers the true shaft radius on a bare cylinder despite
the offset/tilt; it still recovers the shaft radius on the spiny cylinder (rays miss the
spines); and an isotropic (whole-sphere) sampler on that same spiny mesh comes out visibly
inflated — the effect the perpendicular restriction is correcting for. No CAVE/mesh-download
dependency, so this is a standalone correctness check, not something that needs `sbatch` beyond
whatever environment has `trimesh`/`meshparty` installed.

## Gotchas already hit, so they aren't hit again

- **Never fetch a whole-neuron mesh.** `data/neuron_mesh.py::fetch_local_mesh_patch` uses
  `cv.mesh.get(root_id, bounding_box=...)`, which CloudVolume's graphene mesh source restricts
  server-side to just the L2 fragments intersecting the box — confirmed live (5734 vertices for
  a ~12µm-wide patch, not a multi-million-face whole mesh). Nearby vertices are grouped into
  `DEFAULT_BUCKET_SIZE_NM=80µm` spatial buckets (`data/build_dendrite_thickness.py`) so one
  patch fetch covers several vertices instead of one fetch per vertex.
- **Bbox units.** `Bbox(..., unit="nm")` is local bookkeeping only — CloudVolume's graphene mesh
  source expects **voxel** coordinates in the manifest request regardless of what `unit=`
  claims. Always `.convert_units("vx", resolution=cv.resolution)` before calling `mesh.get()`.
  `cv.resolution` for minnie65's segmentation source is `[8, 8, 40]` nm/voxel — not the `(4,4,40)`
  convention some other CAVE services use, so read it from the live `CloudVolume` instance
  rather than hardcoding it.
- **`cv.mesh.get()` doesn't return a `trimesh.Trimesh`.** It returns CloudVolume's own
  lightweight `Mesh` (plain vertices/faces/normals, no `.ray` accessor). `neuron_mesh.py`
  converts explicitly before handing it to `dendrite_thickness.py`'s embree-backed ray casting.
- **HTTP connection-pool starvation.** `cloudfiles.CloudFiles`' mesh-fragment fetch path
  defaults to `num_threads=20` (independent of `CloudVolume(..., parallel=1)`, which only
  governs volumetric chunk downloads), exceeding `requests`' default `pool_maxsize=10` —
  every burst past 10 discarded and reopened a connection, a full fresh TCP+TLS handshake each
  time. `neuron_mesh.py::_raise_connection_pool_size` patches
  `requests.adapters.DEFAULT_POOLSIZE` up before any CloudVolume/GCS client is constructed.
  Fixed alongside raising `DEFAULT_BUCKET_SIZE_NM` (fewer, larger patches → fewer total
  fetches): ~250s/cell → ~104s/cell.
- **No shared on-disk mesh-fragment cache.** Tried once (`CloudVolume(..., cache=...)`
  pointing at a shared directory, to dedupe the overlapping margins of adjacent buckets) and it
  hung a full production run with zero progress for 8+ minutes across every shard: SLURM array
  tasks land on *different compute nodes*, all locking the same cache directory over NFS —
  a classic hang source. Reverted (`cache=False`, CloudVolume's default) rather than debugged
  further; the connection-pool + bucket-size fixes above already gave sufficient throughput.
