#!/bin/bash
#SBATCH --job-name=chk_cable
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/chk_cable_%j.out
#SBATCH --error=logs/chk_cable_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"; mkdir -p logs
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$SLURM_JOB_ID"; mkdir -p "$NUMBA_CACHE_DIR"
PYTHONPATH="$REPO" "$REPO/segclr_db/.venv/bin/python" -u scripts/check_cable_neighborhoods.py
