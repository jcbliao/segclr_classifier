#!/bin/bash
# Bulk dendrite-thickness ingestion, as a SLURM ARRAY -- per explicit user
# direction (2026-08-07). Each array task owns a fixed, weight-balanced
# shard of cells (data/build_dendrite_thickness.py::balanced_shards, keyed
# on manifest n_nodes_covered as a proxy for expected ray-casting work per
# cell) and is independently resumable, so a preempted/timed-out task only
# ever loses ITS OWN shard's progress, not the whole run.
#
# --requeue lets SLURM automatically resubmit a task that gets preempted
# (mit_preemptable jobs can be preempted by higher-priority partitions) or
# node-failed, rather than marking it failed outright. Safe here because
# every cell's .npz is only written after that cell fully completes
# (data/build_dendrite_thickness.py's resumability contract) -- a requeued
# task just re-scans its shard's still-todo cells and continues, no partial
# state to corrupt.
#
# Sizing (array=0-7%4, --workers 4): measured throughput after the
# connection-pool-size + bucket-size fixes (data/neuron_mesh.py,
# data/build_dendrite_thickness.py) was ~104s/cell average on a single
# worker thread (job 19887291, 2 real cells). 8 shards x 4 workers, capped
# at 4 CONCURRENT shards (%4, so max 16 concurrent worker threads CAVE-
# facing at once) keeps aggregate request concurrency in the same rough
# ballpark as the single-process 8-worker test already validated, rather
# than multiplying it by 8x just because the array has 8 tasks -- tune ARRAY/
# WORKERS/THROTTLE via env vars if real throughput turns out to tolerate more.
#
# Configure via env vars, e.g.:
#   LIMIT=2 sbatch scripts/sbatch/build_dendrite_thickness.sh          # validation pass (array of 1)
#   sbatch scripts/sbatch/build_dendrite_thickness.sh                  # full corpus, default sharding
#   ARRAY_SPEC=0-15%4 WORKERS=2 sbatch scripts/sbatch/build_dendrite_thickness.sh   # retune
#SBATCH --job-name=build_dendrite_thickness
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

export CAVE_TOKEN=$(jq -r .token ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json)

# SLURM_ARRAY_TASK_COUNT isn't set on older SLURM (< 19.05); fall back to
# parsing SLURM_ARRAY_TASK_MAX+1 (0-indexed) if it's missing, so this script
# doesn't silently misbehave on a cluster where that var isn't populated.
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX + 1))}"

ARGS=(--workers "${WORKERS:-4}" --shard-index "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS")
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

"$PY" -u data/build_dendrite_thickness.py "${ARGS[@]}"
