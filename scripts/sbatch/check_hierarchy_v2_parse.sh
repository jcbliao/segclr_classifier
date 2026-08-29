#!/bin/bash
# Validates the transcribed hierarchy_v2 tree against the store and steps all
# three sweep configs. GPU per CLAUDE.md's rule that model code runs on a GPU
# node; mit_preemptable because quicktest has no GPUs. Runs in seconds, so
# preemption risk is irrelevant.
#SBATCH --job-name=check_hierarchy_v2_parse
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_hierarchy_v2_parse.py
