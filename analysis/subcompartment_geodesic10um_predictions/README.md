# Five-fold geodesic 10-µm compartment predictions

Scope: 2,223 neurons and 127 astrocytes (2,350 unique roots). Original TEASAR
and CAVE geometry for both; presynaptic points for neurons only. Microglia,
oligodendrocytes and OPCs are excluded. All five trained ResNet checkpoints are
used independently, including training/test/not-used cells with their fold links.
The training soma exclusion is not applied as an inference mask.

Frozen inputs, model hashes and per-cell routes are in `plan.json` under the
output directory listed in `jobs.json`. Original TEASAR node positions are
checked exactly against the corresponding preserved prefix of the earlier
dense geometry before reusing any SegCLR vectors; predictions index the original
TEASAR names and node IDs. There are no predictions on the interpolated nodes.
CAVE vectors come from the existing HDF5 export explicitly named for SegCLR run
20260603_150412/checkpoint_e0_s95000. Repeated HDF5 rows from five duplicated
astrocyte labels are collapsed by node ID only after checking vector agreement
within 1e-6 float32 rounding. The first row is retained; there is no pooling.
Missing vectors, if any, are generated at the requested geometry's own positions
with the same SegCLR checkpoint and the existing parallel crop pipeline.

Each cached file contains five N×4 raw-logit arrays, node IDs, the hierarchy's
class order, and a frozen-plan hash. Synapse files also contain external synapse
IDs and exact positions. Import uses `add_cell_predictions` and creates five
separate prediction runs, each linked to its classifier checkpoint and CV split.
`agg_spec_id` and `window_nm` are null. Uncertainty is categorical entropy in nats.
The importer also idempotently imports neuronal point geometry/embeddings when
necessary. It never labels a predicted node as ground truth.

Scheduling: the GPU pilot waits for neuronal presynaptic embeddings (25310954),
then array 25316210 runs at most four tasks simultaneously on mit_normal_gpu.
Import 25316213 depends on inference and is held because the shared runs registry
is currently unreadable. It checks registry access before making any changes.
After the registry owner repairs access, release it with
`scontrol release 25316213`. Existing presynaptic import 25310957 is independent.

Validation: six tests cover raw logits/class order, invalid output rejection,
five-fold CAVE/original-TEASAR/presynaptic DB round trips, fold registration,
absence of aggregation, retry idempotence, and duplicate-export handling. A
three-source pilot using all five real models passed in the production Python
environment on CPU. The queued GPU pilot must pass before production inference.

## Batched inference update

Completed shards 0–7 from array 25316210 are preserved. Its pending shards 8–15
were replaced by array 25324240, at four concurrent GPUs on mit_normal_gpu.
The held database import 25316213 now depends on the replacement array.

The new reader fetches original/dense coordinate columns and embeddings for
batches of cells, retaining exact original-node checks without decoding edges or
Python coordinate lists. Two loader threads prefetch bounded batches while the
GPU evaluates all five independent models through torch.func/vmap. Two writer
threads compress results while the next batches are inferred. Results retain
the original frozen-plan hash and per-cell format; no averaging is introduced.

Nine tests pass. GPU benchmark 25324056 processed 518,058 nodes across five folds
in 14.04 seconds (excluding startup and result writing), checking all 24 files
against earlier outputs: logits match within rtol/atol 3e-5 and all predicted
labels match. Peak batch-job RSS was approximately 3.3 GB. Runtime on the full
remaining cohort will also include asynchronous compression and filesystem I/O.

## Astrocyte coverage correction

Array 25324240 stopped when some astrocytes had no entries in the pinned named
TEASAR embedding table. An audit found all neuronal TEASAR embedding cache files
present, but 107 astrocytes absent from the selected named-embedding cache.
The original implementation had generated 51 original-geometry backfills;
the fast rewrite failed to reuse those or generate the others. It is corrected
to reuse the saved backfills and lazily generate only absent original-node
vectors, using the same pinned SegCLR checkpoint and parallel crop pipeline.
There are 56 remaining astrocytes, totaling 1,541,774 nodes. CAVE embeddings and
neuronal presynaptic-point predictions do not require this backfill.

Eleven tests now pass, including incomplete batched coverage, original-node
backfill, cache reuse without regeneration, and preservation of existing vectors.
GPU gap pilot 25332160 validates a previously failing rank-8 batch; resumed array
25332174 (ranks 8–15, concurrency 4) depends on that pilot. Completed per-cell
results are reused. Replacement import job 25332176 is held behind that array;
the earlier dependent import was cancelled after the failed array. Registry
permissions are still blocked and require the owner's repair.
