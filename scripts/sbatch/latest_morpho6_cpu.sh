#!/bin/bash
#SBATCH --job-name=latest_morpho6_inputs
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
source scripts/sbatch/latest_morpho6_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
segclr_db/.venv/bin/python -m py_compile scripts/latest_morpho6_training.py scripts/latest_segclr_features.py scripts/conf0_presynaptic_deployment.py
exec segclr_db/.venv/bin/python -u scripts/latest_morpho6_training.py "$@"
