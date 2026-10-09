#!/bin/bash
# Raw SegCLR mean-pooling baseline on exact-unique CAVE presynaptic windows.
# Position and LPE are intentionally excluded from this baseline.
#SBATCH --job-name=presyn_cave_mean
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs

exec "$PY" -u scripts/train_gnn.py \
  --dataset presynaptic_cave \
  --architecture mean \
  --num-embeddings 10 \
  --epochs "${EPOCHS:-16}" \
  --resume \
  --amp \
  --class-balance sample \
  --batch-size "${BATCH_SIZE:-4096}" \
  --attention-budget "${ATTENTION_BUDGET:-14450688}" \
  --presynaptic-cell-cache "${CELL_CACHE_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-5}"
