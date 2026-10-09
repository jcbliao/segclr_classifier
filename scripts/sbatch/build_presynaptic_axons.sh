#!/bin/bash
#SBATCH --job-name=presyn_axons
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs

MODE=${MODE:-build}
OUT=${OUT:-/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10}
case "$MODE" in
  prepare)
    export CAVE_TOKEN="$(jq -r .token ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json)"
    "$PY" -u data/build_presynaptic_axon_database.py --prepare --out "$OUT"
    ;;
  build)
    NUM_TASKS=${NUM_TASKS:?set NUM_TASKS to the array size}
    "$PY" -u data/build_presynaptic_axon_database.py --build --out "$OUT" \
      --task-id "${SLURM_ARRAY_TASK_ID:-0}" --num-tasks "$NUM_TASKS"
    ;;
  finalize)
    "$PY" -u data/build_presynaptic_axon_database.py --finalize --out "$OUT"
    ;;
  *) echo "MODE must be prepare, build, or finalize" >&2; exit 2 ;;
esac
