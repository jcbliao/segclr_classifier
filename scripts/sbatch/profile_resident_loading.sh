#!/bin/bash
#SBATCH --job-name=profile_resident
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=node3006
#SBATCH --output=logs/profile_resident_%j.out
#SBATCH --error=logs/profile_resident_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD"
segclr_db/.venv/bin/python -m pytest -q scripts/test_packed_presynaptic.py
exec segclr_db/.venv/bin/python -u scripts/profile_resident_loading.py
