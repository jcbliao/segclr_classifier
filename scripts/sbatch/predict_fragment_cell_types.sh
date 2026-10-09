#!/bin/bash
#SBATCH --job-name=fragment_cell_types
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
stage="${1:-infer}"
if [ "$#" -gt 0 ]; then shift; fi
exec segclr_db/.venv/bin/python -u scripts/predict_incoming_fragment_cell_types.py "$stage" \
  --rank "${SLURM_ARRAY_TASK_ID:-0}" --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
