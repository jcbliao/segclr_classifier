#!/bin/bash
#SBATCH --job-name=cave_v1dd_train
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task=${SLURM_ARRAY_TASK_ID:?}
fold=$((task / 6))
conditions=(no_v1dd with_v1dd)
architectures=(mean pointwise_mlp graph_transformer)
condition=${conditions[$(((task % 6) / 3))]}
architecture=${architectures[$((task % 3))]}
base=/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_v1dd_addition
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_cave --presynaptic-database "$base" \
  --manifest "$base/manifests/$condition/fold${fold}.json" \
  --results-dir "results/presynaptic/cave_v1dd_addition/$condition" \
  --architecture "$architecture" --num-embeddings 10 \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 8 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5 --seed 0
