#!/bin/bash
#SBATCH --job-name=casey_k17_summary
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
exec segclr_db/.venv/bin/python scripts/summarize_casey_confidence_k17.py
