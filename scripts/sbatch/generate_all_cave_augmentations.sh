#!/bin/bash
#SBATCH --job-name=cave_aug_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/orcd/scratch/orcd/013/jcbliao/torchinductor_cache/segclr_static_b128_fp16
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/generate_all_cave_augmentations.py \
  --plan "$PLAN" --batch-size 32 --workers 4 --variants-per-forward 4 --compile --amp fp16
