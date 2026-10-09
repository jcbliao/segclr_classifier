#!/bin/bash
# Heavy inference + physical-feature joins for the lightweight analysis notebook.
# Usage: sbatch scripts/sbatch/build_feature_prediction_cache.sh RUN_NAME [RUN_NAME ...]
# With no names, builds every completed fixed-node run that is not already cached.
#SBATCH --job-name=feature_corr
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs analysis/all_windows/feature_prediction_cache
# PYTHONPATH="$REPO", as every other wrapper here sets: sys.path[0] is the
# SCRIPT's directory (analysis/), not the repo root, so the module's own
# `from data.dataset_lcpn import ...` has nothing to resolve against.
PYTHONPATH="$REPO" "$PY" -u analysis/all_windows/feature_prediction_correlation.py \
  --num-workers 15 "$@"
