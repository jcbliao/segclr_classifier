#!/bin/bash
#SBATCH --job-name=v1dd_batch
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export V1DD_PACKED_WRITE_BUFFER_MIB=${V1DD_PACKED_WRITE_BUFFER_MIB:-64}
workers=${V1DD_WORKERS_PER_JOB:-1}
if [[ "$workers" -gt 1 ]]; then
  export V1DD_WORKERS_PER_JOB=1 V1DD_REQUEUE_FROM_PARENT=1
  status=0
  srun --ntasks="$workers" --cpus-per-task=1 --cpu-bind=cores --kill-on-bad-exit=0 \
    bash scripts/sbatch/prepare_v1dd_batch_queue.sh "$@" || status=$?
  if [[ "$status" -eq 75 ]]; then
    scontrol requeue "${SLURM_JOB_ID:?}"
    exit 0
  fi
  exit "$status"
fi
failures=0
while true; do
  status=0
  segclr_db/.venv/bin/python -u scripts/prepare_v1dd_batch_queue.py "$@" || status=$?
  if [[ "$status" -eq 0 ]]; then exit 0; fi
  if [[ "$status" -eq 75 ]]; then
    if [[ "${V1DD_REQUEUE_FROM_PARENT:-0}" -eq 1 ]]; then exit 75; fi
    scontrol requeue "${SLURM_JOB_ID:?}"
    exit 0
  fi
  failures=$((failures + 1))
  if [[ "$failures" -gt 5 ]]; then
    echo "Worker failed repeatedly; stopping after five automatic retries (status=$status)" >&2
    exit "$status"
  fi
  echo "Restarting worker after error status=$status retry=$failures/5" >&2
  sleep 10
done
