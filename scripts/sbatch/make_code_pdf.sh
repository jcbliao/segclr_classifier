#!/bin/bash
# Typesets the project source tree into gnn/code_pdf/gnn_classifier_source.pdf.
# CPU-only, runs in seconds; quicktest is plenty.
#SBATCH --job-name=make_code_pdf
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"

"$REPO"/segclr_db/.venv/bin/python -u scripts/make_code_pdf.py "$@"
