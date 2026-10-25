#!/bin/bash
#SBATCH --job-name=cave_segmask_full
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=node3006
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/home/jcbliao/.conda/envs/segclr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/013/jcbliao/torchinductor_cache/segmask_b32_fp16_n6
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/generate_segmentation_augmentations.py infer \
    --plan "${PLAN:?PLAN is required}" --batch-size 32 --workers 4 --compile --compile-morphology
