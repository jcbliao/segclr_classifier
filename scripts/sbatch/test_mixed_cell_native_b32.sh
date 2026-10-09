#!/bin/bash
# Controlled b32 batching test: same windows and optimizer-step count as the
# original run, with windows from roughly 16 cells mixed into each batch.
#SBATCH --job-name=presyn_mixed_b32
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%2
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) architecture=mean ;;
  1) architecture=graph_transformer ;;
  2) architecture=pointwise_mlp ;;
  *) exit 2 ;;
esac
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_dense \
  --presynaptic-database /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/training_local_center/scale32/k9 \
  --results-dir results/presynaptic/new_skeletons_native/mixed_cell_b32/scale32/k9 \
  --architecture "$architecture" --num-embeddings 9 \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 4 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
