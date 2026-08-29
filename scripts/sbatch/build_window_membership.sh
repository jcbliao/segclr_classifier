#!/bin/bash
# Radius is set with WINDOW_NM, e.g.
#   WINDOW_NM=40000 sbatch scripts/sbatch/build_window_membership.sh
# Each radius writes its own data/window_membership*/ cache and skips cells
# already present there, so a rerun is cheap and radii never overwrite one
# another. --mem 64G rather than 32G because window size grows with radius:
# a 40um window holds several times the nodes a 10um one does.
#SBATCH --job-name=build_window_membership
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u data/build_window_membership.py --window-nm "${WINDOW_NM:-10000}"
