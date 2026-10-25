# Packed training cache retention

User instruction (2026-10-25): caches can be deleted after use.

Treat packed training windows as temporary. Remove a packed cache after its
last consuming training/evaluation job has completed successfully; check for
other consumers when several configurations share a cache key. Do not retain
completed-run packed caches solely for possible future reuse.

Implemented by `training_cache_session` in `data/packing_cache.py`, used by
`scripts/train_gnn.py:main`. Shared file locks protect active trainers and CPU
builders. Successful training removes its caches when no other active consumer
holds a lease; failed runs retain them for resume. CPU-only cache preparation
retains output for training. Queued cache-only consumers may rebuild a cache
that was intentionally removed after earlier training completed.

This policy applies to derived packed-window caches, not unique SegCLR
embeddings, source geometry, model checkpoints, manifests, or saved results.

Cleanup inventories record the exact cache directories removed. On 2026-10-25,
132 older caches (~1.8 TiB) and 20 completed confidence-zero caches (~55 GiB)
were removed. All 20 confidence-zero training tasks in array 26829238 were
verified COMPLETED before removing the remaining caches.
