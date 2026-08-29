#!/bin/bash
# Record the learned gamma_0 (global attention) / gamma_1 (adjacency bias) weights
# of every trained GraphTransformer's best checkpoint over the real test windows.
# gamma is predicted per node from its hidden state, so this needs a forward pass,
# not a weight dump -- and the probe recomputes each layer's attention twice
# (with and without the bias) to report what the bias actually moves. GPU-bound,
# so an H200; 16 CPUs feed 15 extraction workers with one core left for the driver.
#
#   LIMIT_BATCHES=2 sbatch scripts/sbatch/check_gamma_weights.sh
#   RUNS="gnn_lcpn_scratch_gt_L4_H4" sbatch scripts/sbatch/check_gamma_weights.sh
#SBATCH --job-name=check_gamma_weights
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

ARGS=(
  --batch-size "${BATCH_SIZE:-2048}"
  --num-workers "${NUM_WORKERS:-15}"
)
# Deliberate word splitting: run directory names contain no spaces.
# shellcheck disable=SC2206
[[ -n "${RUNS:-}" ]] && RUN_LIST=(${RUNS}) && ARGS+=(--runs "${RUN_LIST[@]}")
[[ -n "${LIMIT_BATCHES:-}" ]] && ARGS+=(--limit-batches "${LIMIT_BATCHES}")

echo "running: scripts/check_gamma_weights.py ${ARGS[*]}"
"$PY" -u scripts/check_gamma_weights.py "${ARGS[@]}"
