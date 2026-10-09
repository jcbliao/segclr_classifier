#!/bin/bash
#SBATCH --job-name=casey_k17_folds
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier

confidences=(0 0.3 0.5 0.7 0.9)
architectures=(mean pointwise_mlp graph_transformer)
task=${SLURM_ARRAY_TASK_ID:?}
fold=$((task / 15))
within_fold=$((task % 15))
confidence=${confidences[$((within_fold / 3))]}
architecture=${architectures[$((within_fold % 3))]}
database="/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17/conf${confidence}/fold${fold}"
results="results/presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/conf${confidence}/fold${fold}"

exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_dense \
  --presynaptic-database "$database" \
  --results-dir "$results" \
  --architecture "$architecture" --num-embeddings 17 \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 8 \
  --presynaptic-memmap-root /orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_memmap/scale16/k17 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
