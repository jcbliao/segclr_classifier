#!/bin/bash
#SBATCH --job-name=casey_cave_folds
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
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
if [[ ! "$task" =~ ^[0-9]+$ ]] || ((task > 74)); then
  echo "Invalid array task: $task (expected 0 through 74)" >&2
  exit 2
fi
fold=$((task / 15))
within_fold=$((task % 15))
confidence=${confidences[$((within_fold / 3))]}
architecture=${architectures[$((within_fold % 3))]}
database="/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_confidence_cave/k10/conf${confidence}/fold${fold}"
results="results/presynaptic/casey_confidence_cave/k10/conf${confidence}/fold${fold}"

exec segclr_db/.venv/bin/python -u scripts/train_gnn.py \
  --dataset presynaptic_cave \
  --presynaptic-database "$database" \
  --manifest "$database/manifest.json" \
  --results-dir "$results" \
  --architecture "$architecture" --num-embeddings 10 \
  --epochs 30 --resume --amp --class-balance sample \
  --batch-size 4096 --attention-budget 14450688 \
  --mixed-cell-batch-size 16 --presynaptic-cell-cache 16 --num-workers 8 \
  --gt-depth 4 --gt-heads 4 --lr 1e-4 --weight-decay 1e-5
