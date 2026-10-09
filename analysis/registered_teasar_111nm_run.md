# Batched preparation restart — 2026-09-11T16:20:47

Replaced per-cell writes and readbacks under a global lock with bounded batches
(up to 8 cells or 1 million accumulated raw+dense nodes). Arrow conversion and
full raw/dense array readback occur outside the commit lock. Existing identities
are checked under the lock; retries cannot duplicate rows. Completed preparation
markers are reused. Generic small-fragment compaction is deferred in prep workers
by raising their process-local threshold; the million-row target is unsuitable
for these large nested skeleton rows during ingestion.

Validation: isolated-store append/readback, idempotent retry, and conflicting
geometry rejection passed. Production pilot verified its first batches in 11.1
and 13.0 seconds. After other workers started, 143 additional preparation markers
appeared in 44 seconds (priority 155 -> 298); this is an early throughput sample,
not a completion-time guarantee. Current batch logs have no failure markers.

Active prep: 22600085 (priority rank 0), 22600102 (priority ranks 1–31, concurrency
16), 22600103 (remaining, concurrency 8). The full prep arrays were released after
production pilot batches passed. Priority inference still waits for both priority
prep jobs. Remaining inference waits for remaining prep and priority verification.

Old prep jobs were canceled; Slurm canceled their dependent inference jobs too.
Rebuilt the chain preserving the 4 preemptable / 2 normal GPU split:

```
stage	job_id	partition	concurrency	dependency
priority_preemptable	22600106	preemptable	4	afterok:22600102:22600085
priority_normal	22600107	normal	2	afterok:22600102:22600085
priority_verify	22600108	cpu	1	afterok:22600106:22600107
remaining_preemptable	22600109	preemptable	4	afterok:22600108:22600103
remaining_normal	22600110	normal	2	afterok:22600108:22600103
remaining_verify	22600111	cpu	1	afterok:22600109:22600110
```

Earlier snapshot follows (superseded job IDs):

# Registered TEASAR inference, 2026-09-11

Checked 2026-09-11T16:04:04 (cluster local time).
Run directory: `/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911`.
Frozen cohort: 2,442 cells; routes.json records skeleton names and storage paths.

Pilot 22599395 passed: 32 nodes, 64 dimensions, named-storage roundtrip.
Preparation arrays 22599396 (priority, concurrency 16) and 22599397 (remaining,
concurrency 8) are running without failure markers in current logs.

- priority: 60/2209 prepared
- remaining: 10/233 prepared

Priority GPU arrays 22599427 (96 tasks, concurrency 4, preemptable) and
22599428 (48 tasks, concurrency 2, normal GPU) await priority preparation.
Verification 22599430 follows both arrays.
Remaining GPU arrays 22599431 (16 tasks, concurrency 4, preemptable) and
22599463 (8 tasks, concurrency 2, normal GPU) await remaining preparation and
priority verification. Final verification: 22599466.

Dependencies and throttles checked in Slurm; pilot dependency has cleared.
No resubmission needed. GPU arrays have not started yet. Completion reports:
priority_completion.json and remaining_completion.json in the run directory.
