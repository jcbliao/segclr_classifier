#!/bin/bash
#SBATCH --job-name=v1dd_post_batch
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
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export V1DD_PACKED_WRITE_BUFFER_MIB=${V1DD_PACKED_WRITE_BUFFER_MIB:-64}
export V1DD_POST_CROP_THREADS=${V1DD_POST_CROP_THREADS:-8}
for attempt in 1 2 3 4 5 6; do
 status=0
 segclr_db/.venv/bin/python -u scripts/prepare_v1dd_post_batch_queue.py "$@" || status=$?
 if [[ "$status" -eq 0 ]]; then exit 0; fi
 if [[ "$status" -eq 75 ]]; then scontrol requeue "${SLURM_JOB_ID:?}"; exit 0; fi
 sleep 10
done
echo 'Worker retries exhausted; requeueing instead of leaving unfinished work abandoned' >&2
scontrol requeue "${SLURM_JOB_ID:?}"
exit 0
