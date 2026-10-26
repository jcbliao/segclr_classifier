#!/bin/bash
#SBATCH --job-name=pre_geo20_prepare
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
segclr_db/.venv/bin/python -m pytest -q scripts/test_presynaptic_radius.py scripts/test_training_config.py
exec segclr_db/.venv/bin/python -u scripts/prepare_presynaptic_radius.py --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
