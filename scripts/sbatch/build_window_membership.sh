#!/bin/bash
#SBATCH --job-name=build_window_membership
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u data/build_window_membership.py
