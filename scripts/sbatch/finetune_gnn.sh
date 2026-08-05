#!/bin/bash
# Classification training/fine-tuning of the GNN. GPU job.
# Configure via env vars, e.g.:
#   MODE=finetune PRETRAINED_CKPT=results/pretrain_random/checkpoint_e99.pt sbatch scripts/sbatch/finetune_gnn.sh
#SBATCH --job-name=finetune_gnn
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

MODE="${MODE:-scratch}"
ARGS=(--mode "$MODE" --epochs "${EPOCHS:-100}" --depth "${DEPTH:-2}")
if [[ -n "${PRETRAINED_CKPT:-}" ]]; then
  ARGS+=(--pretrained-ckpt "$PRETRAINED_CKPT")
fi

"$PY" scripts/finetune_gnn.py "${ARGS[@]}"
