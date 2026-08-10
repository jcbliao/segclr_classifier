#!/bin/bash
# Re-check segclr_db's cell_labels/label_hierarchies/splits state now that
# the store's permissions were opened further (2026-08-06). Read-only,
# cheap (single-table scans over ~2-3k rows) -- CPU partition.
#SBATCH --job-name=explore_db_labels
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/explore_db_cell_labels.py
