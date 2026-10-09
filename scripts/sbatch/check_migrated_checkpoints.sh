#!/bin/bash
# Rebuild every results/ checkpoint's model from its own stored config and load
# its weights -- the guard on the architecture renames having reached inside
# the checkpoints, not only the paths.
#
# CPU-only: no forward pass, no data, no GPU. Seconds per checkpoint.
#SBATCH --job-name=check_migrated_ckpts
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"

PYTHONPATH="$REPO" "$PY" -u scripts/check_migrated_checkpoints.py
