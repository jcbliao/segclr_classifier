#!/bin/bash
#SBATCH --job-name=cave_emb_aug_gt
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier

conditions=(clean gray flip structured_low)
task=${SLURM_ARRAY_TASK_ID:?}
condition=${conditions[$((task % 4))]}
fold=$((task / 4))
aug_database=${AUGMENTATION_DATABASE:-/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_embedding_training_choices/conf0.7/all_folds/max_fp16}

exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_cave \
  --presynaptic-database /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10 \
  --manifest /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold${fold}/manifest.json \
  --embedding-augmentation "$condition" \
  --embedding-augmentation-database "$aug_database" \
  --results-dir results/presynaptic/cave_embedding_augmentation_conf0.7 \
  --architecture graph_transformer --num-embeddings 10 \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 8 --num-workers 8 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
