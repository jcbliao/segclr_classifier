# Shared segclr_db import

The four configurations are `cave_n10` and `single_pre_post`, each with
three-class and six-class heads and five folds (20 classifier checkpoints).
Their frozen Parquet deployment contains 160,190,532 included predictions.

The 2026-10-25 inventory used upstream commit `cfb6a8f`. PR #4
(`2b76b44`, K-nearest aggregation and the explicit v7 registry extension)
has merged. The shared store already has the extension: schema drift is
empty, and the migration preview reports no required changes. The two
learned-pooling aggregation specs and our 20 prediction runs were absent.
The October 16 compatibility assessment is historical on these points.

Run the importer from the repository root through Slurm:

```bash
sbatch scripts/sbatch/import_presynaptic_cell_types.sh preflight
# After successful preflight:
sbatch scripts/sbatch/import_presynaptic_cell_types.sh import
```

The wrapper uses a separate upstream worktree in
`logs/segclr_import_upstream`, leaving the existing editable dependency
checkout intact. It must exist at the pinned commit before submission.

Preflight verifies all checkpoint and manifest hashes, all part provenance
metadata, exact class ordering, biological synapse identities and both cell
roots against the shared catalog, and every included skeleton anchor against
`nodes` with node set `cave_skeleton_v4`. All 20 folds are exercised in a
temporary v7 store, checking labels, uncertainty, float32 logits and synapse
identities exactly, including interrupted-write recovery and duplicate-free
retries. It writes `logs/presynaptic_import/preflight.json`; `passed` must be
true before import can run.

Import registers per-fold classifier experiments, runs and checkpoints, the
two learned-pooling specs, and explicit deployment-specific prediction runs.
Experiment specs retain frozen model configuration, checkpoint hashes,
manifest hashes, training arguments, head topology, class order, postsynaptic
input semantics and selection policy. The run timestamp records the import
registration time in UTC, not an asserted historical training launch.
Unknown checkpoint step is recorded as -1. Unverified label-version and
node-label provenance is not fabricated as training-data registry links.

Saved scores are repartitioned into 32 groups of whole target cells. Writes proceed by target group and classifier fold. Each group is compared
against both prediction tables before and after writing. An incomplete group
is removed and replayed only within that deployment's prediction run, node
set and disjoint target roots. A single-importer filesystem lock prevents
concurrent copies of this importer. Restart with the same command and state.
The final check verifies every model/fold's total labels and per-level logits.

`logs/presynaptic_import/progress.json` is live progress. Only
`complete: true` means the import finished. A successful final receipt is
written to `cell_types/segclr_db_import.json` beside the original artifacts.
The frozen October 8 audit and plans remain unchanged as historical records.
The original Parquet files remain the full record of joint probabilities;
conditional head logits require the registered head topology to decode them.

The temporary-store tests passed on October 25 for all 20 folds. All
8,028,116 synapses included by at least one fold matched both biological cell
roots in the shared synapse catalog. The full frozen hierarchy content hashes
also match all three shared Casey hierarchies. The initial node preflight
encountered three owner-only index files owned by `collina`; the current
preflight disables scalar indexes and scans node data directly. No shared
file permissions were changed.

Full preflight job `26827446` passed: zero missing/mismatched synapses and zero
missing anchor nodes among 7,662,984 distinct anchors. Source ranks split target
cells, so the importer stages the required saved score columns by a deterministic
target hash into 32 disjoint groups. Every target occurs in exactly one group.
Import job `26827474` depends on that successful preflight.

Checkpoint sources are inside the private home directory. The importer copies
weights, frozen manifests and training-result JSON into group-readable,
content-addressed directories at `cell_types/models/<checkpoint-sha256>/` and
verifies every copy's hash before registration. Checkpoint registry paths point
to these shared immutable copies. A revised preflight/import pair supersedes
the first import submission to include this artifact publication step.

Revised preflight `26827958` passed with shared checkpoint publication included
in the importer. Import `26827969` supersedes `26827474`, which was cancelled
before any group completed. These are the current job IDs; the progress JSON
and final receipt determine completion.

Import `26827969` is running. All 20 fold artifacts were published to shared
storage and registered. Initial production groups have passed exact read-back
checks. Completion is still pending; consult `progress.json` and the final
receipt rather than treating successful submission as a completed import.
