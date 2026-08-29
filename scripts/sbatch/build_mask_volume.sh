#!/bin/bash
# Per-node SegCLR mask volume, as a SLURM ARRAY. Each task owns a node-balanced,
# contiguous shard of cells (data/build_mask_volume.py::plan_shards) and is
# independently resumable -- a cell's npz is written only after that cell finishes, via
# a tmp-then-rename, so a preempted task leaves no partial state and just re-scans.
#
# CPU-ONLY, deliberately. This reads the sharded precomputed segmentation off ORCD disk
# and reduces each 129^3 crop to one integer; there is no model and no GPU work, so a GPU
# allocation would sit idle. That also means it does not compete with training jobs for
# the gpu=4 QOS pool.
#
# Nothing is downloaded: the segmentation is local (/orcd/compute/sdorkenw/001/collina/
# minnie_seg_1718_sharded), read through TensorStore. No CAVE token, no network.
#
# The work is local-disk I/O plus gzip decompression of 256x256x32 uint64 chunks, both of
# which release the GIL -- so threads within a task and tasks within the array both scale.
# --num-threads is the per-task knob; the inference pipeline measured throughput falling
# past 32 threads with EM reads competing, and this path reads only the segmentation.
#
# The array size is a submit-time argument, not an env var -- SLURM reads --array before
# the script runs, so it cannot be set from inside it. Shard count comes from
# SLURM_ARRAY_TASK_COUNT, so --array is the only place to change the split:
#
#   LIMIT=5 sbatch scripts/sbatch/build_mask_volume.sh              # pilot, the default 1 task
#   sbatch --array=0-31 scripts/sbatch/build_mask_volume.sh         # full corpus, 32 shards
#   THREADS=8 sbatch --array=0-63 scripts/sbatch/build_mask_volume.sh
#
#SBATCH --array=0
#SBATCH --job-name=build_mask_volume
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
# 64G, not 32G: at 32G every shard peaked at exactly the cap and 26 of 32 were OOM-killed.
# The chunk cache is back at its benchmarked 500 MB (see data/mask_volume.py), which is the
# actual fix; this is headroom on top of it, matching the ~6G/cpu the embedding pipeline
# gives the same read path.
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err

set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PYTHON="$REPO/segclr_db/.venv/bin/python"

THREADS="${THREADS:-16}"
LIMIT_ARG=""
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

cd "$REPO"

echo "host        $(hostname)"
echo "array task  ${SLURM_ARRAY_TASK_ID:-none} / ${SLURM_ARRAY_TASK_COUNT:-1}"
echo "threads     ${THREADS}"

# -u because Python block-buffers stdout when it is not a TTY, which would leave a live
# job looking silent for many minutes.
exec "$PYTHON" -u data/build_mask_volume.py \
    --num-threads "${THREADS}" \
    ${LIMIT_ARG} \
    "$@"
