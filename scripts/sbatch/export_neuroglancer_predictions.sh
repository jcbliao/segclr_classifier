#!/bin/bash
# GPU inference/cache validation and immutable export-plan preparation. Use
# scripts/submit_neuroglancer_predictions.sh for the complete array workflow.
#SBATCH --job-name=ng_predictions
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
mkdir -p logs /orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/segclr_predictions/fold_test_validation \
  /orcd/scratch/orcd/013/jcbliao/segclr/window_prediction_cache
"$PY" -u scripts/export_neuroglancer_predictions.py \
  --phase prepare --num-workers 15 "$@"
