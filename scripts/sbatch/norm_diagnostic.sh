#!/bin/bash
# Embedding-norm diagnostic over the already-built local dataset. Informative
# only -- see scripts/norm_diagnostic.py docstring. Pure numpy/scipy/torch CPU
# work over cached files, so mit_quicktest is plenty.
#SBATCH --job-name=norm_diagnostic
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" scripts/norm_diagnostic.py
