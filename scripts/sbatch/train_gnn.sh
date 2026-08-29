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
# Two-hour segments improve backfill eligibility. Runs that need longer
# continue from checkpoint_last.pt when requeued/resubmitted.
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
# Fixed 10/20/40-node sets have predictable memory use, so the entire sweep,
# including GT, runs on the standard L40S queue.
#SBATCH --job-name=train_gnn
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --account=mit_general
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
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
  --epochs "${EPOCHS:-16}"
  --num-workers "${NUM_WORKERS:-31}"
  --resume
  --gt-depth "${GT_DEPTH:-4}"
  --gt-heads "${GT_HEADS:-4}"
  --mpnn-layers "${MPNN_LAYERS:-2}"
  --num-embeddings "${NUM_EMBEDDINGS:-20}"
  --batch-size "${BATCH_SIZE:-4096}"
)

# Ablation switches go through EXTRA_ARGS, word-split deliberately so a caller
# can pass several (EXTRA_ARGS="--gt-no-lpe --gt-no-rel-pos").
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=(${EXTRA_ARGS})

echo "running: scripts/train_gnn.py ${ARGS[*]}"
"$PY" -u scripts/train_gnn.py "${ARGS[@]}"
