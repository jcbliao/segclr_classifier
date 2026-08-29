#!/bin/bash
#SBATCH --job-name=dump_nuc
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/dump_nuc_%j.out
#SBATCH --error=logs/dump_nuc_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"; mkdir -p logs
PYTHONPATH="$REPO" "$REPO/segclr_db/.venv/bin/python" -u scripts/dump_nucleus_positions.py
