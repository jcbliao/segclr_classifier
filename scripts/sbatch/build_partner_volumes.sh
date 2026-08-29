#!/bin/bash
#SBATCH --job-name=partner_vol
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err

set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_CONF=background_thread:false

PY="$REPO/segclr_db/.venv/bin/python"
if [[ "${MERGE:-0}" == "1" ]]; then
  "$PY" -u data/build_partner_volumes.py --merge
else
  "$PY" -u data/build_partner_volumes.py --request-workers "${REQUEST_WORKERS:-4}" "$@"
fi
