#!/bin/bash
# Partition, account, QoS, and array range supplied at submission.
#SBATCH --job-name=cleaned_meshes
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/cleaned_meshes_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/cleaned_meshes_%A_%a.err
set -euo pipefail
REPO=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mesh_matplotlib_${SLURM_JOB_ID}"
exec /orcd/home/002/jcbliao/.conda/envs/segclr/bin/python -u scripts/rebuild_cleaned_mesh_layer.py \
  --roots "$ROOTS" --output "$OUTPUT" --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$WORKERS"
