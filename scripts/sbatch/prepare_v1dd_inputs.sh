#!/bin/bash
#SBATCH --job-name=v1dd_inputs
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
segclr_db/.venv/bin/python -u scripts/infer_v1dd_candidates.py --prepare-inputs \
  --rank "${SLURM_ARRAY_TASK_ID:-0}" --world "${SLURM_ARRAY_TASK_COUNT:-32}" --max-seconds 42000 "$@" || status=$?
if [[ "$status" -eq 75 ]]; then
  # Resume under the same job ID so downstream afterok dependencies stay valid.
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
exit "$status"
