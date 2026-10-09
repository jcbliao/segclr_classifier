# V1DD candidate SegCLR inference

The shared database now has a separate `v1dd` dataset at
`/orcd/compute/sdorkenw/001/segclr-db/v1dd`, schema v5, scoped to
`v1dd_public` materialization 1196. Use the schema-v5 package already used by
the named-skeleton pipelines:

```bash
export PYTHONPATH=/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
```

```python
from segclr_db import SegCLRDatabase
db = SegCLRDatabase(root='/orcd/compute/sdorkenw/001/segclr-db', dataset='v1dd')
```

The cohort includes all 266 cells with `status_axon=true` from the fine-table
ITC, DTC, and STC-Neurogliaform- labels: 71 putative bipolar, 173 putative
Martinotti, and 22 neurogliaform cells. No dendrite-status condition is applied.
All ITC subtypes map to BipFam according to the user's VIP grouping policy;
all DTC subtypes map provisionally to MartFam. Those are broad candidate pools,
not visually confirmed morphologies. The `v1dd_fine` label set preserves the
original labels; the `cell_type` label set records these provisional families.

Inference embeds every node of each CAVE v4 skeleton with the existing
`resnet_860b_reshuffled__20260603_150412 / checkpoint_e0_s95000` 64-dimensional
checkpoint. All 266 skeletons passed availability checks. Model execution uses
`torch.compile(mode='reduce-overhead', dynamic=False)`, FP16 autocast, and
exactly 128 inputs on every invocation. The final batch is padded and its
padding outputs discarded. OOM does not silently reduce the requested batch.
The existing compilation cache is reused at
`/orcd/scratch/orcd/013/jcbliao/torchinductor_cache/segclr_static_b128_fp16`.

Remote V1DD EM mip 3 and segmentation mip 2 both have 38.8 x 38.8 x 45 nm
voxels; crops are 129 cubed. This differs from the MICrONS training resolution
of 32 x 32 x 40 nm, following the existing V1DD inference convention of the
nearest available scale. Segmentation is downloaded as supervoxels and masked using the immutable
leaf membership of each v1196 root. Membership is fetched once per cell and
cached on disk. Three sampled masks matched the original timestamp-based
root lookup exactly. Sixteen crop readers share a 2 GiB in-memory cache of decoded EM chunks
and binary cell masks. Sixteen separate chunk workers download missing chunks
concurrently; a future per chunk prevents duplicate downloads and mask work.
Shared disk volume caching is disabled
because concurrent cache writes produced a decompression error in a pilot.
Batch logs report crop wait times and GPU forward times. Missing data and empty masks fail
explicitly; partial cells are not committed as complete.

Artifacts, cohort, manifest and status files:
`/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196`.
Embeddings are saved per root as NPZ and committed to the v1dd database.
CPU verification checks all node IDs, finite vectors and exact database readback.

Submitted on September 30, 2026:

- CPU skeleton preparation: 24474388 (resumes cached skeletons).
- L40S GPU pilot: 24474426 (tests 129 nodes across full and padded batches).
- GPU array: 24474603, four shards, at most four running; depends on successful
  preparation and pilot.
- CPU verification: 24474619, runs after the GPU array terminates.

Submission does not establish inference completion. Check `pilot_status.json`,
`inference_status_*.json`, `verified_summary.json`, and SLURM/log status.

Code: `scripts/infer_v1dd_candidates.py`,
`scripts/sbatch/prepare_v1dd_candidates.sh`, and
`scripts/sbatch/infer_v1dd_candidates.sh`.

September 30 performance update: sampled image and mask crops matched the
original implementation exactly. On an allocated GPU node, two successive
128-crop batches took 18.70 and 14.37 seconds (6.84 and 8.90 crops/s),
including uncached chunks. Repeating a cached batch took 0.49 seconds
(263 crops/s). These are crop-loading rates, not end-to-end model rates.
Running shards were requeued under the existing array ID, preserving saved
cells and downstream dependencies; concurrency increased from two to four.
The array wall-time allowance was increased to 24 hours. Compilation, FP16,
fixed batches of 128, and the existing compilation cache remain configured.


## CPU chunk prefetch

Shared persistent cache: `v1dd_candidates_v1196/chunks_v1196_v1`.
Image chunks are decoded NumPy arrays; per-root binary mask chunks are packed
into bits before saving. Both use an atomic temporary-file replacement and
4096 shared flock stripes. GPU and CPU producers follow the same protocol,
so concurrent misses cannot expose incomplete files or duplicate a download.
The in-memory 2 GiB cache remains in front of this shared disk cache.

CPU prefetch runs without torch/model loading and follows the same Morton
order as inference (verified against the inference implementation). It skips
completed cells and resumes existing chunks. GPU fallback downloads use the
same cache and locks; GPUs do not depend on all prefetch jobs finishing.

Submitted September 30:
- Normal CPU array 24487568: 16 shards, 8 concurrent, 8 CPUs and 32 GiB each,
  12-hour limit, graceful stop after 42000 seconds.
- Quicktest CPU array 24487569: 4 shards, 4 concurrent, 8 CPUs and 32 GiB each,
  15-minute limit, graceful stop after 600 seconds; warms each GPU shard first.
- Running GPU shards were requeued under array 24474603 to load this version.
  Saved cells and downstream classification dependencies are preserved.

Disk cache validation matched an original image and root mask exactly. With
remote downloads explicitly forbidden and the RAM cache cleared, reconstructing
that crop from disk took 0.039 seconds. This single-crop check is not an
end-to-end inference throughput estimate. CPU progress is recorded in
`logs/v1dd_prefetch_<array>_<task>.out` and `prefetch_status_<array>_<task>.json`.
Wrapper: `scripts/sbatch/prefetch_v1dd_chunks.sh`.


## Prepared batch handoff and worker comparison

The CPU stage now writes atomic input batch directories under `inputs_v1/<root>`:
uint8 image arrays, bit-packed masks, and exact original CAVE node IDs. Each
batch contains at most 128 rows; the GPU still pads the last batch to exactly
128. A cell-level `ready.json` is published only after every batch exists.
The GPU uses double-buffered input reads with no CloudVolume/CAVE client in
prepared-input mode. Three sampled prepared images and masks matched the
original crop loader exactly.

A short L40S benchmark on four prepared batches (512 embeddings per trial)
measured 24596, 32185, and 32105 embeddings/min after compilation warm-up.
These are short prepared-input trials, not whole-cohort sustained throughput;
CPU input preparation remains a separate cost. Results are recorded in
`v1dd_candidates_v1196/prepared_input_benchmark.json` and
`logs/v1dd_prepared_input_benchmark.out`.

The old normal chunk-prefetch array 24487568 was replaced by CPU input
preparation array 24488906 (32x8 CPUs). On restoring network access, a
five-minute production sample measured 1584.2 ready inputs/min with all 32
workers running. It was then replaced by 24490429 (64x4 CPUs, 16 GiB per task,
256 total CPUs and the same aggregate memory allocation). Existing chunks and
complete input batches are reused. GPU array 24474603 has been requeued with
`--prepared-inputs` and depends on successful completion of 24490429; downstream
verification and classification dependencies retain their job IDs.

`analysis/v1dd_cell_types/worker_benchmark` records per-configuration rates and
running worker counts. This is a comparison of successive production intervals;
cache coverage and cell workload vary. Helpers:
`scripts/measure_v1dd_input_rate.py`, `scripts/benchmark_v1dd_input_workers.sh`,
and `scripts/sbatch/prepare_v1dd_inputs.sh`.

The five-minute 64x4 sample measured 1816.6 ready inputs/min, versus 1584.2
for 32x8 (14.7% higher). Both samples had every requested worker running.
64x4 is retained. The array was requeued to enable automatic resumption under
the same job IDs at the 42000-second soft preparation limit; complete batches
are preserved and actual errors still fail the jobs. Results are in
`worker_benchmark/comparison.csv`.

## Per-cell cache lifecycle (September 30)

Scratch reached its 1,000,000-entry quota. The legacy shared image cache was removed; existing root-specific masks and atomic prepared batches were retained. New image and mask chunks are stored in `chunks_v1196_v1/cells/<root>/chunks.sqlite`, one SQLite database per cell. Per-thread connections use SQLite transactions and existing lock stripes; existing legacy masks are reused on demand.

CPU preparation deletes the cell database and legacy root masks only after all prepared batches and `ready.json` are saved. It drains readers and closes cache connections first. A CPU worker then waits for inference to consume that cell before starting another. GPU workers scan their pending roots for ready cells instead of depending on completion of the entire CPU array. Compiled FP16 fixed-128 inference is unchanged. Prepared inputs are removed only after complete embeddings are atomically saved, validated, and committed to the SegCLR database. Incomplete work remains available on restart; soft deadlines requeue the same job IDs.

Current CPU array: **24492777** (8 workers, 4 CPUs/16 GB each). GPU array: **24492778** (4 shards, at most 2 concurrent L40S GPUs). Verification: **24492783**, after both arrays succeed. Classification dependencies were restored: topology 24492796, windows 24492797, finalize 24492798, mean/MLP 24492799, GT 24492800, summary 24492801.

Validation: concurrent requests reused one cached image per cell; image and packed-mask round trips matched exactly; releasing one cell preserved another cell; active-reader cleanup was rejected. Python compilation and shell syntax checks passed. A new scratch-file creation probe succeeded after legacy image cache removal.

## Shared storage reservations and expanded preparation

Replacement CPU array **24493245** has 64 workers, four CPUs/16 GB each. Before admission, each worker reserves its cell's complete prepared-input footprint plus its exact unique image/mask chunk footprint (including SQLite overhead/headroom). Reservations are protected by a shared flock and atomic JSON updates in `storage_budget.json`. The V1DD pipeline cap is **600 GiB**, initialized against **207.39 GiB** of existing pipeline data. Initial accounting includes old partial inputs, masks, cell databases, and other pipeline files, so existing data does not fall outside the cap. Other scratch data is approximately 309 GiB, leaving roughly 100 GiB of account headroom. This budget covers this pipeline; unrelated jobs can still increase their own scratch usage.

Reservations persist over restarts. Completing preparation releases chunk space and retains the prepared-input reservation; GPU consumption releases the remaining space only after cleanup succeeds. Workers can start more cells as space becomes available rather than waiting for their previous cell to be consumed. GPU workers now use a shared ready-cell queue with nonblocking per-cell locks and commit markers, avoiding the blocked-shard problem when only two GPUs are allocated. Array **24492778** remains the GPU array; verification **24492783** now depends on it and the replacement CPU array. Downstream classification job IDs remain unchanged.

Validation: eight concurrent processes could reserve only five 20-unit cells under a 100-unit budget. A sixth admission waited; cleanup enabled admission; ready-state updates shrank reservations; delayed CPU cleanup did not resurrect a GPU-released reservation. Projected unique-chunk counts matched the production crop geometry, including clipped boundary chunks. Python compilation and shell syntax checks passed.

## GPU dispatch per completed cell

Dispatcher **24493970** now submits a separate L40S job when each cell's complete prepared inputs appear (20-second polling). Jobs request four CPUs, 48 GB, and 30 minutes; they use the existing compiled FP16 fixed-128 cache. Root-specific logs and status files avoid collisions. The submission ledger `gpu_cell_jobs.json` and existing per-root inference locks prevent duplicate work, with active-job recovery and up to three attempts for failed cell jobs. Already-saved embeddings can be recovered and committed without recomputing vectors. Dispatcher completion requires all 266 commit markers. Verification **24492783** now depends on CPU array **24493245** and dispatcher **24493970**; persistent GPU array 24492778 was cancelled. Downstream classification dependencies remain attached to verification.

Dispatcher tests covered ready-only submission, saved-vector recovery, duplicate suppression, and complete-cohort gating. Two real cell jobs were submitted (24493978 and 24493979), and the first commit marker was observed.

A fresh scratch scan measured about **373 GiB outside V1DD** (presynaptic_axons 243 GiB, skeletons 109 GiB, neuroglancer 13 GiB, embedding_paths 8 GiB, and smaller caches). The prior 309 GiB estimate used a stale report. A 900 GiB V1DD-only cap would exceed the 1 TiB account quota unless other scratch data is moved. The current 600 GiB pipeline cap remains pending clarification of the requested 900 GB cap.

## Chunk last-use reclamation and batch prefetch

CPU array 24493245 was requeued under the same ID to deploy exact chunk-lifetime tracking. `v1dd_chunk_lifetimes.py` enumerates only the storage chunks intersecting the requested crops, records first/last use in Morton-ordered 128-node batches, and calculates peak simultaneous chunk-plus-prepared-input storage. Reservations now use this peak instead of the whole-cell accumulated cache. Legacy masks, existing ahead-of-use SQLite chunks, abandoned temporary batches, and one-time cache migration workspace remain conservatively accounted for.

After an input batch is atomically published (or a saved batch is validated), chunks whose last use is that batch are removed from RAM, SQLite, and legacy mask files. SQLite databases gain pointer maps once at migration. A drained-batch DELETE transaction uses FULL auto-vacuum to physically reclaim all free pages, then restores INCREMENTAL mode for chunk reads/writes. Testing found that this Python/SQLite build reclaimed just one page per incremental-vacuum PRAGMA; deleting rows alone was insufficient. Shared/future-use chunks remain available, and active readers block eviction.

Batch prefetch schedules the exact unique image/mask chunks for all 128 crops together. Shared futures pin their results through crop assembly, including chunks evicted from the ordinary RAM LRU. This batches scheduling and eliminates duplicate within-batch loading; remote storage still serves individual object requests. The bounding box of a whole batch is not downloaded, so unrelated intervening chunks are not fetched.

Validation in `input_profile/test_chunk_reclamation.py`: exact geometry and first/last use, shared-chunk preservation, resume/skip idempotence, actual file shrinkage, reader guards, reduced shared reservations, and exact image/mask reconstruction with batch prefetch. Checks passed. Real cell 864691132619338714 (15,673 nodes) fell from 89.24 GiB projected accumulated storage to 35.80 GiB peak storage, with 1.60 GiB peak live chunks (before legacy/migration allowances). Production logs confirmed successive batches reclaiming 402.52, 379.66, and 432.31 MiB in one resumed cell. The budget admitted 18 cells versus 11 before this rollout. Sustained throughput has not yet been remeasured.

## October 1 morning status and sparse-segmentation recovery

At 08:45 EDT, 132/266 cells had complete commit markers. Three cells completed in the preceding hour; zero prepared cells were waiting for GPUs. Scratch quota report (08:36) showed 706.2/1024 GB and 248.1K/1M files. Shared pipeline reservations were near their 600 GiB cap, with 22 cells admitted.

CPU tasks 38, 45, and 46 failed on omitted segmentation storage chunks (`EmptyVolumeException`, 256x256x32 chunks). Background-only sparse segmentation chunks now become zero masks; missing image chunks still raise, and every assembled crop must retain a nonempty target-cell mask before preparation can succeed. Focused checks passed for both cases. Because Slurm had already retired the failed tasks, replacement array **24541051** runs these ranks explicitly with `--world 64` to preserve their original root assignments. Verification 24492783 now waits for original CPU array termination (`afterany`), successful recovery array completion, and dispatcher 24493970 success. Dispatcher success still requires all 266 validated cell commits; final verification checks every vector against the shared DB.

## Additional preemptable CPU array

User requested another 64 CPU jobs on `mit_preemptable`. Array **24541785** was submitted with 64 tasks, four CPUs/16 GB each, account `mit_general`, QoS `normal`, 12-hour time limit, and requeue enabled. It shares the existing 600 GiB V1DD storage budget, chunk last-use reclamation, prepared-batch reuse, and per-cell GPU dispatcher.

`cpu_assignment.json` now partitions the cohort into 128 disjoint worker positions: original normal array 24493245 and recovery array 24541051 occupy ranks 0–63; the new preemptable array occupies ranks 64–127. Existing CPU arrays were requeued to load the updated assignment before the new array was released. New preparation cell locks guard the complete per-cell lifecycle across restarts and preemption. The assignment check proved disjoint coverage of all 266 cells and that the new assignments of already-completed original shards are subsets of their completed work.

Verification 24492783 now waits for original-array termination, successful recovery and preemptable arrays, and successful dispatcher completion. Job IDs for subsequent windows and classifiers remain unchanged.

### 2026-10-01 shared batch queue
Replaced static whole-cell preparation arrays 24493245/24541051/24541785 with cooperative arrays 24544106 (64 mit_normal workers) and 24544107 (64 mit_preemptable workers), 4 CPUs/16 GiB each. Each worker claims an unfinished 128-input batch; there is no per-cell concurrency limit. Durable per-batch file locks recover abandoned claims on preemption. Existing complete batches and old chunk caches are reused. Shared chunks use 64 SQLite shards per cell and cross-process single-download locks; reclamation waits for all dependent batches to be durable. The shared 600 GiB budget conservatively reserves full cell chunk and prepared-input storage for unrestricted batch ordering. GPU dispatcher 24493970 continues unchanged; verification 24492783 now depends on the new arrays ending and dispatcher success.

Validation: multiprocess distinct batch claims (including >8 on one cell), abandoned-claim recovery, out-of-order completion/finalization, cross-process chunk reuse and exact last-use deletion tests passed. Existing image/mask reconstruction and physical reclamation tests passed. Python compilation and wrapper shell syntax passed. Queue initialization preserved existing completed work and reconciled approximately 341.8 GiB of existing V1DD data. Throughput after this switch remains to be measured.

Added preemptable array 24544561 (64 workers) at user request, bringing the preemptable pool to 128 submitted workers alongside 64 normal workers. All use the same batch queue and shared storage budget. Verification dependency includes the added array.

Added preemptable array 24544852 (128 workers) at user request, doubling the submitted preemptable pool from 128 to 256 workers (4 CPUs/16 GiB each). Shared batch queue and 600 GiB storage budget unchanged. Updated verification dependency to include this array before releasing it.

### Whole-cell cache cleanup and worker recovery
At user request, disabled all per-batch SQLite DELETE/auto-vacuum operations in cooperative workers. Entire cell caches are removed only after every prepared batch is durable and its readers have closed their connections; prepared inputs remain until GPU commit. This also lets complete cells finish without opening damaged cache databases during cleanup. Workers restart their Python process after exceptions, up to five automatic retries per job execution with 10-second backoff; planned time limits requeue and scheduler preemption remains requeue-enabled. Replacement arrays 24545731 (64 normal) and 24545737 (256 preemptable) replace 24544106/24544107/24544561/24544852; verification dependencies updated. Existing queue, complete batches and storage reservations preserved. Python compilation, shell syntax and concurrency tests passed. The prior 10-minute speed measurement overlaps this intervention and cannot be treated as a steady-state measurement of the revised implementation.

### Confirmed corrupt cache cleanup
Held and requeued all cooperative workers, waiting for running jobs to stop before auditing 978 SQLite files in unfinished cells with `PRAGMA quick_check(1)`. Five corrupt shared shards in roots 864691132809153825 (44), 864691132800779776 (39,33,34), and 864691132783228812 (22) were removed with their sidecars, freeing 0.464 GiB. Completed prepared-input batches were preserved. Other databases passed. Audit details: input_profile/corrupt_cache_cleanup_20261001.json. Added an exclusive file lock around each shared shard's reads/writes, covering the complete access operation across threads/nodes. No incremental deletion remains. Multi-process shared-cache/queue tests and Python compilation passed. Released worker arrays 24545731 and 24545737 to reconstruct missing shards. The failures are confirmed to involve newly created shared shards; pre-existing corruption in older per-cell caches did not explain this audit's results.

### Reduce shared-cache and queue contention
Profiling showed 129/150 seconds waiting on the global queue lock before a batch could start. A direct batch sample spent 92.3% of accumulated chunk-task time waiting on shard/chunk locks, 7.3% downloading, with little time in SQLite operations or mask computation. Moved downloads and duplicate-download waits outside the shard lock; all SQLite reads/writes remain guarded by short shard critical sections. Moved expensive candidate-file probes and storage admission checks out of the global queue lock; read-only snapshots no longer rewrite queue JSON, and denied reservations no longer rewrite budget JSON. Global lock now protects only actual state changes. Different workers rotate their starting batch while preserving closest-cell preference and unrestricted per-cell parallelism. Partial-cell collection no longer writes progress for unused per-batch reclamation. Concurrency tests cover simultaneous different-chunk downloads within one shard, exactly one download for duplicate requests, integrity checks, abandoned batch-claim recovery, and successful claims while the global queue lock is held. Compiling Python and wrapper syntax passed. Requeued existing arrays 24545731 and 24545737 to load these changes; no quota/configuration/model changes.

### NFS blocking-lock handover diagnosed
A three-node microbenchmark with 50 acquisitions and 2 ms critical sections took 0.117 s locally but 30.35 s and 60.66 s on the other nodes, nearly all spent in blocking `flock`. Using nonblocking `LOCK_NB` plus jittered retry on the same NFS mount completed in 0.115–0.420 s across all three nodes. Introduced v1dd_file_lock.acquire_lock for cache, queue state updates and storage budget locks: identical exclusive flock semantics and process-exit recovery, avoiding the blocking handover path. Duplicate-download stripes increased from 4096 to 16384 with a new prefix; workers were stopped before switching so old/new ownership never overlaps. Queue claims remain nonblocking and probes outside the global lock. Concurrency/cache integrity/recovery tests and Python compilation passed. Requeued arrays 24545731/24545737 again to activate this fix; throughput measurement follows.

### One-CPU expansion after measured speedup
Polling-lock measurement (poll_lock_rate.json): 42,529 new inputs in five minutes including restart/scheduler startup; final three minute intervals ~14,813, 7,972 and 18,876 inputs/minute; ten cells committed. One-CPU instrumented full batch: 172.943 s preparation, 34.76 CPU-seconds over 184.1 s including setup, peak RSS 2549.8 MiB. CPU remains underutilized. Replaced per-thread eight-entry SQLite connection LRUs with one shared connection per shard, protected by explicit per-process thread locks and cross-node advisory locks; new SQLite cache files use default no-vacuum mode because production cleanup deletes whole cell caches. This avoids ~1469 opens during the profiled batch. Tests added for sixteen crop threads sharing connections, duplicate reuse and integrity, all passed.

Replaced 24545731/24545737 with arrays 24550084 (64 mit_normal) and 24550197 (384 mit_preemptable), each one CPU/6 GiB, maintaining 128-input batches and shared 600 GiB budget. Total 448 preparation workers use 448 allocated CPUs versus the old 320-worker pool's 1280 CPUs. The partition's per-user submit cap is 448, including other preemptable jobs; 384 leaves headroom. Verification 24492783 depends on new arrays and GPU dispatcher 24493970. Saved work/queue/reservations preserved.

### Extra workers trimmed and dispatcher recovery
Attempted user-requested 500 additional one-CPU workers. The normal partition also has a 448 submitted-job cap, so 440 individual normal jobs were rejected. Submitted 24551588 (60 single-worker preemptable jobs) and 24551715 (220 normal jobs with two one-CPU workers and 12 GiB per job). User then requested cancelling excess jobs that would remain queued. Cancelled 82 normal array tasks beyond concurrent account memory capacity, then the remaining 110 additional normal tasks still pending; retained 28 running normal two-worker jobs and all 60 running preemptable jobs. Current preparation pool: 120 normal workers in 92 jobs plus 444 preemptable workers in 444 jobs = 564 running workers; no extra prep jobs pending at audit time.

GPU dispatcher 24493970 failed on a transient nonzero squeue query; Slurm cancelled its dependent verifier/classifier chain. Added retry handling for scheduler command errors and polling lock acquisition to the dispatcher, then restarted as 24552031. Restarted verification as 24552134 after all surviving prep arrays and dispatcher success. Confirmed all 266 topology/site artifacts exist and original topology array completed, so restored only remaining stages: windows 24552372, finalize 24552373, mean/MLP 24552374, GT 24552375, summary 24552376. All have appropriate dependencies. GPU submission/commit progress resumed.

Final steady five-minute measurement (final_pool_rate.json): 66,562 new inputs in 523 batches = 13,286.6 inputs/minute; 14 cells committed. Minute rates ~11,653, 17,434, 13,601, 14,007, 9,696. This is ~9.5x the earlier 1400/minute shared-worker rate, and ~7.3x the earlier 1817/minute static preparation sample. Cohorts/windows differ, so those comparisons are throughput observations rather than matched experimental speedups. A packed-cache prototype remains experimental and has not been enabled for production; its isolated read benchmark and recovery tests assess whether further cache-format changes are justified.

Packed cache deployed 2026-10-01 ~12:09 EDT: preparation imports PackedCrops; all four retained arrays requeued and released. Packed concurrency/recovery tests passed. Five-minute startup-inclusive measurement: 83,861 durable inputs, 16,709 inputs/minute, four cell commits. Last two intervals 25,777 and 23,562/minute, approximate per-worker rates 88 and 68/minute using endpoint-average running worker counts (pool ramping). Reserved storage 538.4 GiB/600 at minute four; dispatcher running. Full measurement: input_profile/packed_pool_rate.json.

Packed write batching for future runs: BufferedPackedCache defaults to 64 MiB per worker (V1DD_PACKED_WRITE_BUFFER_MIB). Process-local reads see buffered data; threshold and batch-close flush group chunks by shard with data fsync before index fsync. Batch publication follows flush; killed workers lose only regenerable unacknowledged cache entries. Cross-worker duplicate fetches are possible during buffering, but publication deduplicates under the shard lock. Tests passed concurrency, deduplication, failure retry, corrupt extent recovery and threshold flushing. Scratch filesystem write-only benchmark, 512 x 256 KiB chunks: individual 2.786s/1024 fsyncs, buffered 1.035s/256 fsyncs, 2.69x write speedup. This is not an end-to-end throughput measurement. Existing live workers retain their loaded code; future submissions/restarts pick up batching. Current near-complete run was not restarted solely for this future-run change.

Packed buffered ownership update: downloaded chunks transfer their striped flock handle into BufferedPackedCache. Successful durable data+index publication releases ownership. Flush when 64 MiB or 256 owned chunks accumulate, at batch close, and before waiting for an occupied chunk stripe; flushing before waiting breaks cyclic dependencies across workers. Cached/migrated reads need no retained download ownership. Process death automatically releases unpublished ownership so the next worker regenerates it. Supersedes the prior warning about cross-worker duplicate downloads from buffered downloads. Tests: opposite-order two-process ownership gives exactly two downloads with no deadlock; killed owner recovered and reused; buffered multiprocess/thread integrity/recovery and existing queue/SQLite tests pass. Changes apply to future submissions/restarted workers; existing loaded workers are unchanged.

User-requested exclusion: root 864691132773444371 rejected by API as invalid at materialization 1196; no batches or embeddings produced. Removed from active cohort (preserved backups and excluded_invalid_roots.parquet), manifest updated to 265 cells: 172 MartFam,71 BipFam,22 NglFam. Queue marked EXCLUDED and reservation released. Discarded temporary remapped leaf membership; no remapped inference performed. Build checks use active manifest/cohort count. Dispatcher and verification/classification dependencies preserved; training arrays 24558222 and24558223 remain split three tasks each.
