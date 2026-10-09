#!/bin/bash
# Carry the architecture renames (fully_connected -> mpnn_complete, deepsets ->
# pointwise_mlp) onto everything already on disk: the run names everything is
# keyed by, the ModelConfig stored inside each checkpoint (which --resume
# compares against the freshly built one), and the state_dict keys that follow
# the renamed submodule.
# One-off; idempotent, so a repeat is a no-op.
#
# Pass --dry-run through to see the plan without moving anything:
#   sbatch scripts/sbatch/migrate_architecture_names.sh --dry-run
#
# File renaming plus small JSON rewrites, and a torch.load/torch.save per
# checkpoint on CPU -- no model is built and nothing trains, so quicktest
# rather than a GPU partition.
#SBATCH --job-name=migrate_arch_names
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

PYTHONPATH="$REPO" "$PY" -u scripts/migrate_architecture_names.py "$@"
