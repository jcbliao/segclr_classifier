#!/bin/bash
#SBATCH --job-name=cell_type_import
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SEGCLR_IMPORT_CHECKOUT=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/segclr_import_upstream
exec segclr_db/.venv/bin/python -u scripts/import_presynaptic_cell_types.py "$@"
