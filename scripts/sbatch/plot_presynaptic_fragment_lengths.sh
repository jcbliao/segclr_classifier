#!/bin/bash
#SBATCH --job-name=presyn_fragment_lengths
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec segclr_db/.venv/bin/python -u scripts/plot_presynaptic_fragment_lengths.py \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" --num-tasks "${NUM_TASKS:-32}" "$@"
