#!/bin/bash
#SBATCH --job-name=teasar_target
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/home/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec segclr_db/.venv/bin/python -u scripts/build_teasar_target.py \
  "${MODE:-prepare}" --task-id "${SLURM_ARRAY_TASK_ID:-0}" --num-tasks "${NUM_TASKS:-32}" "$@"
