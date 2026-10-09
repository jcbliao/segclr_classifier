#!/bin/bash
# Submit the 14 requested ResNet-head models at 10, 20, and 40 embeddings.
#
# ONLY=<architecture> restricts the sweep to one architecture, and the depth of
# whichever one it is comes from MPNN_LAYERS / POINTWISE_MLP_LAYERS / GT_DEPTH
# (2, 2 and 4 by default). Together they submit one depth of one architecture
# without touching the rest of the grid:
#
#   ONLY=graph_transformer GT_DEPTH=2  bash scripts/sbatch/submit_embedding_sweep.sh   # 12 jobs
#   ONLY=mpnn MPNN_LAYERS=4            bash scripts/sbatch/submit_embedding_sweep.sh   # 12 jobs
#   ONLY=pointwise_mlp POINTWISE_MLP_LAYERS=4 \
#                                      bash scripts/sbatch/submit_embedding_sweep.sh   # 3 jobs
#
# ONLY=mpnn matches `mpnn` exactly, so it leaves `mpnn_complete` alone.
#
# The depth is already part of every run name (`mpnn_L{n}`, `pointwise_mlp_L{n}`,
# `gt_L{depth}_H{heads}`), so two depths coexist in results/ rather than
# overwriting each other -- and the analysis parser reads the depth as part of
# the ARCHITECTURE, so they land on separate panels instead of being averaged
# together as folds of one model.
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"

job_index=0
skip_jobs=${SKIP_JOBS:-0}

submit() {
  local n=$1 architecture=$2 extra=${3:-} batch_size cpus workers pool
  # Filtered out before job_index moves, unlike SKIP_JOBS: an ONLY sweep is a
  # sweep in its own right, so its jobs have to be numbered from zero for the
  # preemptable/normal rotation below to stay in the intended 4:2 ratio.
  if [[ -n "${ONLY:-}" && "$architecture" != "$ONLY" ]]; then
    return
  fi
  if (( job_index < skip_jobs )); then
    ((job_index += 1))
    return
  fi
  case "$n" in
    10) batch_size=4096 ;;
    20) batch_size=2048 ;;
    40) batch_size=1024 ;;
  esac
  if [[ "$architecture" == mean ]]; then
    cpus=32
    workers=31
  else
    cpus=16
    workers=15
  fi

  # Sustainable per-user concurrency is four preemptable GPUs and two total
  # normal GPUs. The normal partition's 32-CPU ceiling is shared across
  # accounts, so mit_general and AMF are not additive pools. Feed the queues
  # in the same 2:1 ratio and use the higher-priority AMF association on normal.
  pool=$((job_index % 6))
  common=(--gres=gpu:1 --cpus-per-task="$cpus")
  if (( pool < 4 )); then
    sched=(--partition=mit_preemptable --account=mit_general --qos=normal)
  else
    sched=(--partition=mit_normal_gpu --account=mit_amf_standard_gpu \
           --qos=mit_amf_standard_gpu)
  fi

  NUM_EMBEDDINGS="$n" ARCHITECTURE="$architecture" \
    MPNN_LAYERS="${MPNN_LAYERS:-2}" \
    POINTWISE_MLP_LAYERS="${POINTWISE_MLP_LAYERS:-2}" \
    GT_DEPTH="${GT_DEPTH:-4}" \
    BATCH_SIZE="$batch_size" NUM_WORKERS="$workers" LR="${LR:-1e-4}" \
    WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}" EXTRA_ARGS="$extra" \
    sbatch "${sched[@]}" "${common[@]}" scripts/sbatch/train_gnn.sh
  ((job_index += 1))
}

for n in 10 20 40; do
  submit "$n" mean

  # Set-only learned aggregator over raw embeddings. Pointwise MLP deliberately
  # rejects position/LPE: this is the learned permutation-invariant control.
  submit "$n" pointwise_mlp

  # The same shape with the nonlinearity taken out. A node-wise linear map
  # commutes with the mean, so this is the control that says how much of the
  # pointwise MLP's margin over mean pooling is the learned nonlinearity.
  submit "$n" linear

  submit "$n" mpnn_complete
  submit "$n" mpnn_complete "--position"
  submit "$n" mpnn_complete "--position --lpe"
  submit "$n" mpnn_complete "--lpe"

  submit "$n" mpnn
  submit "$n" mpnn "--position"
  submit "$n" mpnn "--position --lpe"
  submit "$n" mpnn "--lpe"

  submit "$n" graph_transformer "--gt-no-rel-pos --gt-no-lpe"
  submit "$n" graph_transformer "--gt-no-lpe"
  submit "$n" graph_transformer
  submit "$n" graph_transformer "--gt-no-rel-pos"
done
