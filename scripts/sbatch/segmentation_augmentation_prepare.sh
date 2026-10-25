#!/bin/bash
#SBATCH --job-name=cave_segmask_prepare
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jcbliao/.conda/envs/segclr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/generate_segmentation_augmentations.py prepare \
    --plan "${PLAN:?PLAN is required}" --output "${OUTPUT:?OUTPUT is required}" \
    --limit-cells 0 --nodes-per-cell 0 --tasks 18 --radius 3 --neighbors 6 --amp fp16 --variants-per-forward 4 --include-held-out
