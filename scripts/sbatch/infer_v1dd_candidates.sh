#!/bin/bash
#SBATCH --job-name=v1dd_segclr
#SBATCH --partition=mit_preemptable,mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/013/jcbliao/torchinductor_cache/segclr_static_b128_fp16
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
/home/jcbliao/.conda/envs/segclr/bin/python -u scripts/infer_v1dd_candidates.py \
  --rank "${SLURM_ARRAY_TASK_ID:-0}" --world "${SLURM_ARRAY_TASK_COUNT:-1}" --prepared-inputs --cell-lifecycle --shared-queue --storage-budget-gib 600 --max-seconds 84000 "$@" || status=$?
if [[ "$status" -eq 75 ]]; then
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
exit "$status"
