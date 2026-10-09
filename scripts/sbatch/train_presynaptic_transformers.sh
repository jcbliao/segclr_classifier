#!/bin/bash
# Train the two full K=10 presynaptic GraphTransformers as a two-task array:
#   0: CAVE skeleton windows, ordinary GraphTransformer
#   1: exact full new-skeleton windows, GraphTransformerTEASAR
#
# No ablations are passed: both use relative position, per-window LPE,
# adjacency bias, global attention, and SegCLR features.  Dynamic batches obey
# the dense-attention budget rather than forcing every variable-size window
# into a fixed batch shape.
#SBATCH --job-name=presyn_gt
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) DATASET=presynaptic_cave; ARCHITECTURE=graph_transformer ;;
  1) DATASET=presynaptic_new; ARCHITECTURE=graph_transformer_teasar ;;
  *) echo "array index must be 0 or 1" >&2; exit 2 ;;
esac

exec "$PY" -u scripts/train_gnn.py \
  --dataset "$DATASET" \
  --architecture "$ARCHITECTURE" \
  --num-embeddings 10 \
  --epochs "${EPOCHS:-16}" \
  --resume \
  --amp \
  --class-balance sample \
  --batch-size "${MAX_WINDOWS_PER_BATCH:-4096}" \
  --attention-budget "${ATTENTION_BUDGET:-14450688}" \
  --presynaptic-cell-cache "${CELL_CACHE_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --gt-depth "${GT_DEPTH:-4}" \
  --gt-heads "${GT_HEADS:-4}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-5}"
