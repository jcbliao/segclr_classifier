#!/bin/bash
#SBATCH --job-name=extended_k17_build
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${FINALIZE:-0}" = 1 ]; then
  exec segclr_db/.venv/bin/python -u data/build_missing_extended_k17.py finalize
fi
exec segclr_db/.venv/bin/python -u data/build_missing_extended_k17.py build \
  --task-id "${SLURM_ARRAY_TASK_ID:?}"
