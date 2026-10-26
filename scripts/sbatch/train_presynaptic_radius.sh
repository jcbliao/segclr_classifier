#!/bin/bash
#SBATCH --job-name=pre_geo20_train
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --exclude=node3006
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
stage=${1:-train}
if [ "$#" -gt 0 ]; then shift; fi
fold=${SLURM_ARRAY_TASK_ID:-0}
if [ "$stage" = smoke ]; then
  segclr_db/.venv/bin/python -m pytest -q scripts/test_presynaptic_radius.py
fi
segclr_db/.venv/bin/python -u scripts/run_presynaptic_radius.py "$stage" --fold "$fold" "$@"
if [ "$stage" = train ]; then
  exec segclr_db/.venv/bin/python -u scripts/run_presynaptic_radius.py classify --fold "$fold" "$@"
fi
