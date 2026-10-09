#!/bin/bash
#SBATCH --job-name=ng_l2_prepare
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs
exec "$PY" -u scripts/export_neuroglancer_predictions_l2.py --phase prepare "$@"
