#!/bin/bash
#SBATCH --job-name=infer_teasar111
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/infer_teasar111_%A_%a.out
set -euo pipefail
export PYTHONPATH=/home/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/jcbliao/.conda/envs/segclr/bin/python
SCRIPT=/home/jcbliao/rotation/segclr/gnn_classifier/scripts/infer_named_teasar.py
if [ "${1:-}" = '--pilot' ]; then
  "$PY" -u "$SCRIPT" --pilot
else
  "$PY" -u "$SCRIPT" --rank "${SLURM_ARRAY_TASK_ID}" --world 32
fi
