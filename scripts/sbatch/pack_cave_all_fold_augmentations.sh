#!/bin/bash
#SBATCH --job-name=pack_cave_all_aug
#SBATCH --partition=mit_normal,mit_preemptable,mit_normal_gpu
#SBATCH --requeue
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons
exec segclr_db/.venv/bin/python -u scripts/pack_cave_training_augmentations.py \
 --manifest "$base"/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold*/manifest.json \
 --output "$base/cave_embedding_training_choices/conf0.7/all_folds/max_fp16" --workers 8
