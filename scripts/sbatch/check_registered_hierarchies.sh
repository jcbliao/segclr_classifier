#!/bin/bash
# Store-only registry read -- no CAVE calls, so no CAVE_TOKEN needed.
#SBATCH --job-name=check_registered_hierarchies
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python


cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_registered_hierarchies.py
