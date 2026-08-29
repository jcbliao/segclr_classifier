#!/bin/bash
# Read-only, CPU-node validation of the new-skeleton/embedding coordinate join.
#SBATCH --job-name=check_skel_align
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
mkdir -p logs
"$REPO/segclr_db/.venv/bin/python" -u scripts/check_new_skeleton_embedding_alignment.py "$@"
