#!/bin/bash
# Read-only check: does data/graph_cache/*.pt have `pos`, and does the new
# rel_pos feature (data/geodesic_window.py) come out correctly on real data.
#SBATCH --job-name=check_pos_field
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_pos_field.py
