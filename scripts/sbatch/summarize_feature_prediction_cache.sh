#!/bin/bash
# Recompute the notebook's summary tables from feature caches already on disk.
# Usage: sbatch scripts/sbatch/summarize_feature_prediction_cache.sh RUN_NAME [RUN_NAME ...]
# CPU-only: no inference and no feature join, so no GPU is requested -- the
# heavy build path is scripts/sbatch/build_feature_prediction_cache.sh.
#SBATCH --job-name=feature_summary
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs
PYTHONPATH="$REPO" "$PY" -u analysis/all_windows/feature_prediction_correlation.py --summarize-only "$@"
