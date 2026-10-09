#!/bin/bash
# Append the fold tag to every pre-existing run name and everything keyed to
# it. One-off; idempotent, so a repeat is a no-op.
#
# Pass --dry-run through to see the plan without moving anything:
#   sbatch scripts/sbatch/migrate_run_names_to_fold.sh --dry-run
#
# CPU-only file renaming plus small JSON rewrites -- no model, no GPU, seconds
# of work, so quicktest rather than a GPU partition.
#SBATCH --job-name=migrate_fold_names
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"

PYTHONPATH="$REPO" "$PY" -u scripts/migrate_run_names_to_fold.py "$@"
