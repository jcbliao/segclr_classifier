# CAVE pre/post V1DD addition

Fold 0, K=10, no confidence exclusion. Compare mean, pointwise MLP and GT across:
- MICrONS + V1DD training (`with_v1dd`) only.
- `pre_only`, `pre_post`, and matched-width `pre_mean_control`.

All conditions use the same eligible post sites and paired held-out cells. Site
eligibility excludes unresolved roots and empty post masks before presynaptic
window deduplication. The excluded invalid V1DD root is not used.

V1DD post-site inventory: 1,037,698 sites across 265 cells, 254 spatial shards;
27 unresolved roots. Roots and coordinates come from outgoing synapses_v1dd at
materialization 1196. Exact 129³ crops are masked by the postsynaptic root at that
timestamp; SegCLR uses FP16, torch.compile, fixed/padded batch 128 and the existing
compilation cache. Raw image/segmentation chunks use a bounded shared RAM cache.
MICrONS post-site vectors are reused from the verified v1718 inference.

The prepared post cache retains synapse IDs, post coordinates (nm), post root IDs,
eligibility and 64D embeddings, aligned to the CAVE window rows.

Job chain:
- Site preparation: 24562166 (completed).
- V1DD inference: 24562305 (preemptable ranks 0–3), 24562306 (normal GPU ranks 4–7).
- Verify inference: 24562337.
- Join post cache, manifests and loader checks: 24562381.
- Training: 24562383 (4 models preemptable), 24562394 (5 normal GPU).
- Paired domain summaries: 24562395.

Database: `/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_pre_post_v1dd_addition`.
Results: `results/presynaptic/cave_pre_post_v1dd_addition`.

Validation: exact crop axes/padding/root-map/chunk reuse test passed; real V1DD synapse 338873726 produced a 129³ crop with 350 matching post-root voxels. MICrONS coverage audit 24562769 checked all 2,029 MICrONS cells: no missing valid-window site IDs. Cache preparation depends on both this audit and V1DD inference verification.

Shared preparation correction: cancelled remote GPU download arrays24562305/24562306. Default V1DD site entrypoint now refuses remote GPU cropping. Shared CPU preparation arrays24565943 (64 normal workers) and24565944 (128 preemptable workers), one CPU/6GiB each, use the existing BatchQueue, StorageBudget and retained-ownership PackedCrops machinery. Packed chunks store images and root-label volumes mapped at materialization1196, compressed for mixed post roots. Each shard reserves full prepared inputs plus4GiB compressed-cache cap; pipeline budget600GiB. Capped chunks fall back to RAM rather than blocking completion. Cache removed only after whole-shard preparation; inputs deleted after verified GPU output. Dispatcher24565945 submits one GPU job per ready shard, alternating partitions under existing QoS limits. Verify24562337 now depends on dispatcher, existing cache/training/summary dependencies preserved. Multiprocess sharing/deadlock/recovery tests passed.

Default launch command: `bash scripts/launch_v1dd_postsynaptic_pipeline.sh` (refuses duplicate active workers/dispatcher). Tests also exercise CPU batch creation, mask/status alignment, whole-shard cache deletion, verified GPU publication with a deterministic model stub, input deletion and reservation release. Live preparation began and produced a durable batch; no throughput claim is made for this startup interval.

Oct1 recovery audit: 7/254 post shards done,204/8108 CPU batches durable; old492-worker pool shrank to13 running,477failed,2OOM. Main errors were API502 on roots_binary and volume info;4GiB caches filled and disabled disk reuse. Added shared8-request API gate with transient retry/backoff and120s root request timeout; reader volumes reuse info from initialization. Increased shard cache cap to16GiB with all reservations updated under600GiB,8cropthreads for memory. Replacement arrays24603366 (64normal) and24603367 (428preemptable) preserve existing data. Worker wrapper requeues exhausted retries; dispatcher requeues FAILED/TIMEOUT/OOM/NODE_FAIL prep tasks once per minute while OPEN work remains. API health probe succeeded0.62s. Recovery/concurrency/lifecycle tests passed.

Oct 1 target-membership correction: shared post preparation now caches raw supervoxel chunks and batches get_leaves_many requests for target roots (16 roots/request), instead of get_roots for every label in a full storage chunk. Root-membership lists share the bounded packed cache; existing image chunks and prepared batches are retained. Historical-root mask validation on a real V1DD site matched exactly (8,216 voxels); shared preparation lifecycle tests forbid full-chunk get_roots calls and pass. Requeued/released arrays 24603366/24603367 to load the corrected implementation. Sustained throughput after deployment remains to be measured.

Oct 2 restoration: removed the global API concurrency gate entirely, including its NFS slot polling and wrapper environment default. Preparation again builds target-cell masks from root membership and raw segmentation, retaining shared packed caching and transient-error retries. Unrestricted 16-way request concurrency and retry behavior tests pass, as do shared preparation lifecycle tests. Requeued arrays 24603366 and 24603367 to load the ungated code; existing prepared batches/output preserved.

Oct 2 historical-root correction: removed get_leaves_many from preparation. Restored get_leaves per root with retained ownership/shared caching. If a leaf endpoint specifically rejects a retired root as not valid at timestamp, preparation instead resolves only unique crop labels via get_roots at materialization1196; authorization failures still propagate. Tests cover this fallback and forbid full-chunk mapping. Real per-root API probes returned 13 and 12 supervoxels (second lookup0.09s). Worker arrays requeued to deploy; aggregate throughput not yet verified.

Oct 2 crop-local restoration: live stacks showed slow per-root leaf requests holding shared ownership locks. Removed all leaf API calls from post preparation. Exact crop supervoxel labels are mapped at materialization1196, matching the initial V1DD post loader, with shared packed raw chunks retained. Tests forbid leaf calls and validate crop-local timestamp mapping. Real two-site CPU batch published successfully in11.02s after initialization (metadata discovery retried three times). Requeued/released worker pool; sustained aggregate rate still unmeasured.

Oct 2 isolated subset validation (full pool canceled at user request): one normal CPU job24615308 prepared two128-site batches in122.986s and92.022s, total220.913s including initialization;255ok,1empty_mask,0request retry errors. Output shapes, node IDs, statuses and nonempty masks verified. Validation JSON in postsynaptic_site_embeddings_v1dd_v1196/debug_subset_20261002/validation.json. Full preparation arrays24603366/24603367 and dispatcher24565945 remain canceled; downstream dependency chain was canceled by Slurm too. No automatic full relaunch. This proves single-worker correctness/progress, not492-worker scaling.

Oct 2 two-worker troubleshooting: full pool24616549/24616550 and dispatcher24616551 canceled. Root mapping now groups32 crops, deduplicates crop-only labels, serializes root RPCs per worker, flushes buffered chunks before RPCs, disables CAVEclient nested HTTP retries, sets10s connect/30s read timeout, and splits failed4096-label requests to smaller requests before retrying128-label failures. No global API limiter. Tests validate grouped lookup, exact masks, and split ordering. Two-worker job24619174 completed512sites in526s (58.4inputs/min total),509ok/3empty_masks,0API retries/splits; first256sites image/mask/status arrays exactly match the prior single-worker output. Root mapping accounts for~65% of summed batch preparation time; scaling beyond two workers is unverified. Fullpool remains canceled. Record: results/presynaptic/cave_pre_post_v1dd_addition/two_worker_validation_20261002.json.

Oct2 scale-up authorized to500+workers: retained64workerpool, added392preemptable tasks(array24660106) and56normal tasks(array24660108),512total preparation workers. Partitionmit_preemptable MaxSubmitJobsPU448 prevented putting all448newworkers there;438CPUtasks leave room forGPUdispatch.74normalCPUworkers total. Dispatcher recovery config tracks all4arrays. Baseline623746inputs/4874batches/152committed shards recorded in results/presynaptic/cave_pre_post_v1dd_addition/pool512_baseline_20261002.json.
