#!/bin/bash
# Eval-time interventions over trained per-window classifiers. One dataset pass
# is shared across every checkpoint and condition at the selected radius.
# GraphTransformer forwards dominate and are GPU-bound, so this uses an H200;
# 16 CPUs feed the 15 extraction workers while leaving one core for the driver.
#
# Configure via env vars, e.g.:
#   WINDOW_NM=20000 sbatch scripts/sbatch/eval_ablations.sh
#   LIMIT_BATCHES=2 RUNS="gnn_lcpn_scratch_meanpool" sbatch scripts/sbatch/eval_ablations.sh
#   CONDITIONS="identity permute_x rewire_lpe0" sbatch scripts/sbatch/eval_ablations.sh
#SBATCH --job-name=eval_ablations
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

ARGS=(
  --window-nm "${WINDOW_NM:-10000}"
  --batch-size "${BATCH_SIZE:-4096}"
  --num-workers "${NUM_WORKERS:-15}"
)

# These list-valued variables are word-split deliberately so each item reaches
# argparse as a separate value. Run directory names and condition names contain
# no spaces in this repository.
# shellcheck disable=SC2206
[[ -n "${RUNS:-}" ]] && RUN_LIST=(${RUNS}) && ARGS+=(--runs "${RUN_LIST[@]}")
# shellcheck disable=SC2206
[[ -n "${CONDITIONS:-}" ]] && CONDITION_LIST=(${CONDITIONS}) \
  && ARGS+=(--conditions "${CONDITION_LIST[@]}")
[[ -n "${LIMIT_BATCHES:-}" ]] && ARGS+=(--limit-batches "${LIMIT_BATCHES}")

echo "running: scripts/eval_ablations.py ${ARGS[*]}"
"$PY" -u scripts/eval_ablations.py "${ARGS[@]}"
