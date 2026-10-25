#!/bin/bash
#SBATCH --job-name=report_folds
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=node3006
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/report_worker_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/report_worker_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export SEGCLR_PACKING_CACHE_ONLY=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
resume_segment() {
    kill -USR1 "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" || true
    scontrol requeue "${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"
    exit 0
}
trap resume_segment USR1
segclr_db/.venv/bin/python -u scripts/run_report_fold_worker.py "${SLURM_ARRAY_TASK_ID:?}" &
worker_pid=$!
wait "$worker_pid"
