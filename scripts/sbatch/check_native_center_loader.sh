#!/bin/bash
#SBATCH --job-name=check_native_center_loader
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec segclr_db/.venv/bin/python -u scripts/check_native_center_loader.py
