#!/bin/bash
#SBATCH --job-name=pre_geo20_summary
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec segclr_db/.venv/bin/python -u scripts/summarize_presynaptic_radius.py "$@"
