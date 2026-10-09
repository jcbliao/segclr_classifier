# SegCLR testing cohort: fresh TEASAR skeletons, 2026-09-10

## SegCLR inference submissions

GPU pilot `22516710` passed: 32 named TEASAR nodes embedded into 64 dimensions,
with an isolated named-embedding storage roundtrip. Earlier pilots `22516408`
and `22516514` failed on unreadable registry data before inference. The runner
now pins the readable model file directly:
`/orcd/data/sdorkenw/001/collina/segclr_runs/resnet_860b_reshuffled_20260603_150412/checkpoints/checkpoint_e0_s95000.pt`.
It reuses `SegCLRInference` crop loading, Morton ordering, batching and forward
pass, with materialization 1718, tf32, batch 32, and 8 reader threads per GPU.

Only the dense named skeleton `teasar_testing_111nm_20260910` is embedded.
Array `22516890` covers the 196 cells ingested at pilot time, frozen in
`inference_ready_roots.txt`. Array `22516893` covers all 466 after final
skeleton ingestion `22515305` succeeds and the first inference array ends;
it reuses completed per-cell embedding files and retries missing ones.
Both use 32 strided tasks, concurrency 4, one L40S per task, 12-hour limit.

Outputs are atomically staged under the resumed output directory's
`named_embeddings/teasar_testing_111nm_20260910/resnet_860b_reshuffled__20260603_150412/checkpoint_e0_s95000/`.
Each completed cell is also committed and read back from the shared store's
`embeddings/named_node_embeddings/d64.lance`. Its key includes
`skeleton_name`, `root_id`, `node_id`, `run_id`, and `checkpoint_id`.
The new table is declared in the v5 worktree's schema; ordinary CAVE
`node_embeddings` queries do not include it. Use `st.scan(store,
"named_node_embeddings", dim=64, filter=...)` from that worktree to read it.
Per-rank JSON reports and SLURM logs record errors; inference never records
a partial cell as complete. File locks prevent duplicate per-cell work and
serialize database commits across GPUs.

## Resumed at user request

Preparation job `22515291` validated and reused 71 skeletons from the original
archive and 246 from the interrupted run. Only 149 of the 466 cells still
need generation; 208 matching dense outputs were reused too.

Output: `/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910_resumed`.
Its `provenance.json` records the source of each reused cell; existing files
are linked rather than copied or replaced. Arrays `22515297` (0–47) and
`22515298` (48–63), global stride 64, generate missing cells and resample
remaining outputs. Dependent ingestion job `22515305` verifies all 466 cells
under `teasar_testing_20260910` and `teasar_testing_111nm_20260910`.
The 71 original-archive cells were also successfully ingested separately
by job `22514885` under the `teasar_existing*` names (142 verified records).

Earlier run history follows.

## Correction: reuse the existing archive

At the user's direction, fresh generation arrays `22514080`, `22514081`
and their dependent ingestion job `22514099` were cancelled. Their partial
outputs remain on disk but are not used for the replacement run.

Audit `22514826` found that the established 2,335-cell TEASAR archive at
`/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons` contains 71 of
the 466 testing-cohort root IDs. Resampling array `22514840` uses those
existing files directly. Its separate output directory is
`/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_existing_teasar_111nm_20260910`,
including explicit available/missing root lists. The remaining 395 IDs
await identification of another existing source; no new generation is running.

The records below describe the original, now cancelled generation run.

The saved cohort has 466 unique root IDs: 20 cells in each of 23 cell types,
plus 6 OPC cells. The two source manifests are identical:

- `/orcd/data/sdorkenw/001/collina/segclr_downstream_analysis/round_1/cells.csv`
- `/orcd/data/sdorkenw/001/collina/downstream_output_full_nodes/round_1/cells.csv`

Both run configurations specify 20 cells per class, seed 42, materialization
1718. No cells were sampled anew.

Output directory:
`/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910`

It contains the copied `cells.csv`, `roots.txt`, and `provenance.json` with
the source hash and cell-type counts. Raw outputs go in `skeletons/`; dense
outputs and per-cell spacing reports go in `resampled_111nm/`.

Generation uses the established
`~/rotation/skeletonization/scripts/skeletonize_segclr.py` pipeline, with
fresh outputs: fetch mesh, heal, default centered TEASAR, radius merge and
component stitching. There is no change to the default algorithm.

Each original edge is subdivided into `ceil(length_nm / 111)` equal pieces.
Original vertices, short edges, branch points, components and cable geometry
are retained. Edge lengths are at most 111 nm before float32 rounding; this
is densification, not exact uniform branch-wise resampling. Radii at added
vertices are interpolated, not newly measured mesh clearances. Compartments
in the named database skeletons are unknown (0).

SLURM jobs:

- Preparation and validation: `22514001`, completed, 47 tests passed.
- Parallel generation/resampling: `22514080` (indices 0–111, concurrency 64)
  and `22514081` (indices 112–127, concurrency 16), global stride 128.
- Database v5 validation: `22514098`, completed, 50 tests passed.
- Dependent serial ingestion: `22514099`, waits for both generation arrays.
- Shared-store preflight: `22514293`, completed; v5 opened, all 466 default
  skeletons found, named table created successfully.

The previous named-skeleton changes were on schema v3. They were applied
without conflicts to an isolated v5 worktree at
`~/rotation/segclr/segclr_db_named_v5` (upstream commit `8334f46`), leaving
the original working tree intact. The ingestion script imports that worktree
via its `_pkgroot` directory and writes to the shared v5 store at
`/orcd/compute/sdorkenw/001/segclr-db`, dataset `microns`.

Names are `teasar_testing_20260910` and `teasar_testing_111nm_20260910`.
They have independent node IDs and do not replace the CAVE skeletons used by
existing embeddings. `skeleton_version=0` denotes these locally generated
artifacts, not a CAVE skeleton-service version.

Completion is recorded in `ingestion_report.json`, with 932 verified named
skeletons expected and an empty missing list. Ingestion round-trips every
array and checks default node counts. The ingestion job fails if any output
is absent. To resume failed generation, submit the same array indices with
the original `OUT`, `ROOTS`, and `GLOBAL_TASKS=128`, then rerun ingestion;
both generation and identical database writes are resumable.
