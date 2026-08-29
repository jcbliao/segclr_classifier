#!/bin/bash
# Characterize the ~0.5% of nodes whose center voxel does not hold their own root_id.
# Read-only. Reads every node's center voxel across the pilot cells (Morton-ordered so
# the chunk cache serves them), then a full window for a bounded sample of the misses.
#SBATCH --job-name=check_mask_center_misses
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
exec "$REPO/segclr_db/.venv/bin/python" -u scripts/check_mask_center_misses.py "$@"
