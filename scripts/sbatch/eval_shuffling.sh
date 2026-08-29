#!/bin/bash
# Inference-only shuffling/topology ablations over existing best checkpoints.
# Override CHECKPOINTS, CONDITIONS, WINDOW_NM, or OUTPUT as needed.
#SBATCH --job-name=eval_shuffling
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

ARGS=(
  --num-workers "${NUM_WORKERS:-31}"
  --batch-size "${BATCH_SIZE:-4096}"
  --window-nm "${WINDOW_NM:-10000}"
  --output "${OUTPUT:-results/shuffling_ablations.json}"
)

# Both variables are deliberately word-split lists of argparse values.
# shellcheck disable=SC2206
[[ -n "${CHECKPOINTS:-}" ]] && ARGS+=(--checkpoints ${CHECKPOINTS})
# shellcheck disable=SC2206
[[ -n "${CONDITIONS:-}" ]] && ARGS+=(--conditions ${CONDITIONS})

echo "running: scripts/eval_shuffling.py ${ARGS[*]}"
"$PY" -u scripts/eval_shuffling.py "${ARGS[@]}"
