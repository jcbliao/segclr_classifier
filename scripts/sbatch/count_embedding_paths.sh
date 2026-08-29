#!/bin/bash
#SBATCH --job-name=cnt_paths
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/cnt_paths_%j.out
#SBATCH --error=logs/cnt_paths_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"; mkdir -p logs
PYTHONPATH="$REPO" "$REPO/segclr_db/.venv/bin/python" -u scripts/count_embedding_paths.py
