#!/bin/bash
#SBATCH --job-name=cave_segmask_pilot
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jcbliao/.conda/envs/segclr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
py=/home/jcbliao/.conda/envs/segclr/bin/python
plan=${PLAN:-analysis/presynaptic/segmentation_augmentation/pilot_plan.json}
"$py" -u scripts/benchmark_segmentation_augmentation.py --batch-size 32 --variants-per-forward 4 --compile-model
if [ -n "${BASELINE_PLAN:-}" ]; then
    "$py" -u scripts/generate_segmentation_augmentations.py infer --plan "$BASELINE_PLAN" --batch-size 32 --workers 4 --compile
    "$py" -u scripts/generate_segmentation_augmentations.py summarize --plan "$BASELINE_PLAN"
fi
"$py" -u scripts/generate_segmentation_augmentations.py infer --plan "$plan" --batch-size 32 --workers 4 --compile --compile-morphology
"$py" -u scripts/generate_segmentation_augmentations.py summarize --plan "$plan"
"$py" -u scripts/generate_segmentation_augmentations.py pack --plan "$plan"
