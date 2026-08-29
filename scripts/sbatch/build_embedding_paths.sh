#!/bin/bash
#SBATCH --job-name=build_paths
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/build_paths_%A_%a.out
#SBATCH --error=logs/build_paths_%A_%a.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"; mkdir -p logs
NUM_TASKS=${NUM_TASKS:?set NUM_TASKS to the total number of slices}
# Node-local numba cache, per task. Shared on NFS it is written by every
# task at once (ESTALE), and mit_preemptable nodes differ in CPU generation,
# so a cached binary from one node can be an illegal instruction on the next.
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$SLURM_JOB_ID"
mkdir -p "$NUMBA_CACHE_DIR"
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
PYTHONPATH="$REPO" "$REPO/segclr_db/.venv/bin/python" -u data/build_embedding_paths.py \
  --task-id "${SLURM_ARRAY_TASK_ID}" --num-tasks "$NUM_TASKS" "$@"
