#!/bin/bash
#SBATCH --job-name=geometry_confidence_notebook
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PATH="$PWD/segclr_db/.venv/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
exec segclr_db/.venv/bin/python -u scripts/make_skeleton_geometry_confidence_notebook.py
