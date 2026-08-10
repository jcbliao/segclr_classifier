#!/bin/bash
# Validates the dendrite-thickness node feature against real cached data:
# the orig_node_ids join, the window slice, and measured-fraction stats.
# CPU-only -- no model, no GPU. --mem 64G: loads the whole train split's
# cell data, same as a training job's dataset construction.
#SBATCH --job-name=check_edge_length_distribution
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_edge_length_distribution.py
