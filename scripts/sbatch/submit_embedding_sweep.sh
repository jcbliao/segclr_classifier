#!/bin/bash
# Submit the 13 requested ResNet-head models at 10, 20, and 40 embeddings.
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"

job_index=0
skip_jobs=${SKIP_JOBS:-0}

submit() {
  local n=$1 architecture=$2 extra=${3:-} batch_size cpus workers pool
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

  NUM_EMBEDDINGS="$n" ARCHITECTURE="$architecture" MPNN_LAYERS=2 \
    BATCH_SIZE="$batch_size" NUM_WORKERS="$workers" EXTRA_ARGS="$extra" \
    sbatch "${sched[@]}" "${common[@]}" scripts/sbatch/train_gnn.sh
  ((job_index += 1))
}

for n in 10 20 40; do
  submit "$n" mean

  submit "$n" fully_connected
  submit "$n" fully_connected "--position"
  submit "$n" fully_connected "--position --lpe"
  submit "$n" fully_connected "--lpe"

  submit "$n" mpnn
  submit "$n" mpnn "--position"
  submit "$n" mpnn "--position --lpe"
  submit "$n" mpnn "--lpe"

  submit "$n" graph_transformer "--gt-no-rel-pos --gt-no-lpe"
  submit "$n" graph_transformer "--gt-no-lpe"
  submit "$n" graph_transformer
  submit "$n" graph_transformer "--gt-no-rel-pos"
done
