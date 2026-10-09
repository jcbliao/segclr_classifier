#!/bin/bash
#SBATCH --job-name=registered_teasar_prep
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
set -euo pipefail
REPO=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$REPO/segclr_db/.venv/bin/python" -u scripts/registered_teasar_pipeline.py "${MODE:-prepare}" \
  --phase "${PHASE:-priority}" --rank "${SLURM_ARRAY_TASK_ID:-0}" --world "${PREP_TASKS:-32}"
