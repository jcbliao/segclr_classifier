#!/bin/bash
# Validate data/mask_volume_cache against the segmentation and against the independently
# measured dendrite radii. See scripts/check_mask_volume.py for what each check rules out
# -- the point is that a wrong-materialization or shifted-window run produces plausible
# numbers rather than an error, so these checks are the only thing standing between that
# and a silently wrong feature.
#
# CPU-only and read-only. Sampling 200 center voxels per cell is a few thousand tiny
# reads, so quicktest's 15-minute cap is ample for a pilot-sized cache.
#SBATCH --job-name=check_mask_volume
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

exec "$REPO/segclr_db/.venv/bin/python" -u scripts/check_mask_volume.py "$@"
