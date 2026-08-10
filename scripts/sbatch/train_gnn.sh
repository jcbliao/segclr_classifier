#!/bin/bash
# Classification training over per-window local subgraphs
# (data/dataset_windowed.py). GPU job. --mem 64G: the dataset eagerly loads
# every cell in the split into memory for window extraction.
#
# Configure via env vars, e.g.:
#   ARCHITECTURE=mean sbatch scripts/sbatch/train_gnn.sh
#   ARCHITECTURE=mpnn MPNN_LAYERS=2 sbatch scripts/sbatch/train_gnn.sh
#   EXTRA_ARGS="--gt-no-lpe" sbatch scripts/sbatch/train_gnn.sh
#
# 12h walltime: measured ~4 min/epoch, so the 100-epoch default needs 5-7h and
# the earlier 4h cap truncated runs. Well under mit_preemptable's 2-day limit.
#
# mit_preemptable: 2-day walltime cap (vs. mit_normal_gpu's 6h) and generally
# a shorter queue, at the cost of being preemptable.
#
# --requeue + --resume together make that cost small: SLURM puts a preempted
# job back in the queue under the same job id, and train_gnn.py picks up from
# checkpoint_last.pt (model + optimizer + RNG state, written every epoch), so
# an interruption costs at most one epoch instead of the whole run. --resume is
# passed unconditionally because it is a no-op on a fresh run -- with no
# checkpoint_last.pt present it simply starts at epoch 0. The same pairing is
# what lets a 100-epoch run finish on mit_normal_gpu despite its 6h cap:
# resubmit until done, each submission continuing where the last was cut off.
#
# 32 CPUs / 31 workers: window extraction costs real CPU per item, and worker
# scaling stays positive out to 31 even though it is clearly sublinear past 15.
#
# H200 rather than L40S: measured GPU utilization on the GraphTransformer runs
# is high, so these are GPU-bound rather than pipeline-bound. Note the mean and
# mpnn architectures are NOT -- meanpool (a zero-parameter aggregator) ran at
# 175 s/epoch against mpnn's 131 s/epoch on L40S, so both sit at a data-pipeline
# floor and gain nothing here. Override to gpu:l40s:1 for those if H200s queue.
#SBATCH --job-name=train_gnn
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

ARGS=(
  --architecture "${ARCHITECTURE:-graph_transformer}"
  --epochs "${EPOCHS:-100}"
  --num-workers "${NUM_WORKERS:-31}"
  --resume
  --gt-depth "${GT_DEPTH:-4}"
  --gt-heads "${GT_HEADS:-4}"
  --mpnn-layers "${MPNN_LAYERS:-2}"
)

# Ablation switches go through EXTRA_ARGS, word-split deliberately so a caller
# can pass several (EXTRA_ARGS="--gt-no-lpe --gt-dist-bias").
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=(${EXTRA_ARGS})

echo "running: scripts/train_gnn.py ${ARGS[*]}"
"$PY" -u scripts/train_gnn.py "${ARGS[@]}"
