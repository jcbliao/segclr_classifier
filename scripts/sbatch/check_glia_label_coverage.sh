#!/bin/bash
# Store-only label/coverage scan -- no CAVE calls, so no CAVE_TOKEN needed.
# quicktest's 15-min cap is ample: the comparable check_new_cells.py scan runs
# in well under a minute.
#SBATCH --job-name=check_glia_label_coverage
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_glia_label_coverage.py
