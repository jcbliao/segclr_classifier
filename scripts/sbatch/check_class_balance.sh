#!/bin/bash
# Checks the ported class-balance sampler and the drop_labels filtering.
# CPU-only (no model is built), but it loads the whole train split's cell
# graphs into memory the way training does, so it needs real RAM and more
# than quicktest's 15 minutes is not expected.
#SBATCH --job-name=check_class_balance
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_class_balance.py --draw
