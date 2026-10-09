#!/bin/bash
#SBATCH --job-name=dense_presynaptic
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/dense_presynaptic_%A_%a.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/home/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$REPO/segclr_db/.venv/bin/python" -u data/build_dense_presynaptic.py "--${MODE:?set MODE}" \
  --out "${OUT:-/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/dense_teasar_multiscale_v2}" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" --num-tasks "${NUM_TASKS:-1}" "$@"
