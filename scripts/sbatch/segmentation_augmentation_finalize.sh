#!/bin/bash
#SBATCH --job-name=cave_segmask_finalize
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jcbliao/.conda/envs/segclr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
py=/home/jcbliao/.conda/envs/segclr/bin/python
"$py" -u scripts/generate_segmentation_augmentations.py summarize --plan "${PLAN:?PLAN is required}"
"$py" -u scripts/generate_segmentation_augmentations.py pack --plan "$PLAN"
