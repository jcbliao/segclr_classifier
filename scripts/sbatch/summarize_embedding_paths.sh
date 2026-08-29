#!/bin/bash
#SBATCH --job-name=sum_paths
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/sum_paths_%j.out
#SBATCH --error=logs/sum_paths_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"; mkdir -p logs
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$SLURM_JOB_ID"; mkdir -p "$NUMBA_CACHE_DIR"
PYTHONPATH="$REPO" "$REPO/segclr_db/.venv/bin/python" -u scripts/summarize_embedding_paths.py
