#!/bin/bash
# Render every notebook widget state from the real Matplotlib figures.
# CPU-only, but deliberately scheduled because the full state space is large.
# Usage: sbatch scripts/sbatch/export_interactive_figures.sh
# Set PUBLISH_FIGURES=1 to commit and push generated pages after a successful export.
#SBATCH --job-name=export_figures
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs

MPLBACKEND=Agg PYTHONPATH="$REPO" uv run --python "$PY" --with mpld3 \
  python -u scripts/export_mpld3_figures.py

if [ "${PUBLISH_FIGURES:-0}" = "1" ]; then
  git add docs/figures
  if ! git diff --cached --quiet -- docs/figures; then
    git commit -m "Refresh Matplotlib interactive figures" -- docs/figures
  fi
  git push origin HEAD
fi
