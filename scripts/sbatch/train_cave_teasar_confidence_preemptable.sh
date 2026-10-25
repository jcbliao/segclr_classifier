#!/bin/bash
#SBATCH --job-name=morpho_confidence_preempt
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --exclude=node3006
#SBATCH --requeue
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
resume_segment() {
  echo '90-minute segment ending; requeueing from the latest atomic epoch checkpoint.'
  scontrol requeue "${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"
}
trap resume_segment USR1
# A background child lets bash handle the walltime signal while it waits.
segclr_db/.venv/bin/python -u scripts/run_cave_teasar_confidence_sweep.py "${SLURM_ARRAY_TASK_ID:?}" &
trainer_pid=$!
wait "$trainer_pid"
