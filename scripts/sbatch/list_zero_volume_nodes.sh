#!/bin/bash
# Enumerate every node whose SegCLR window contains none of its own cell.
# Read-only; scans mask_volume_cache and reads pos from the graph cache.
#SBATCH --job-name=list_zero_volume_nodes
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
exec "$REPO/segclr_db/.venv/bin/python" -u scripts/list_zero_volume_nodes.py "$@"
