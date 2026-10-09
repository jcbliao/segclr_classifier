#!/bin/bash
#SBATCH --job-name=pack_cave_aug
#SBATCH --partition=mit_normal,mit_preemptable,mit_normal_gpu
#SBATCH --requeue
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
exec segclr_db/.venv/bin/python -u scripts/pack_cave_training_augmentations.py \
  --manifest /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0/manifest.json \
  --workers 16
