#!/bin/bash
# Submit prepare/test -> parallel array -> summary. Arguments pass to all modes.
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
mkdir -p logs
NUM_TASKS="${NUM_TASKS:-16}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
if ! [[ "$NUM_TASKS" =~ ^[1-9][0-9]*$ && "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'NUM_TASKS and MAX_CONCURRENT must be positive integers\n' >&2
    exit 1
fi
export NUM_TASKS
prepare=$(sbatch --parsable --cpus-per-task=2 scripts/sbatch/compare_skeleton_geometry.sh prepare "$@")
prepare="${prepare%%;*}"
array=$(sbatch --parsable --dependency="afterok:$prepare" \
  --array="0-$((NUM_TASKS - 1))%$MAX_CONCURRENT" \
  scripts/sbatch/compare_skeleton_geometry.sh run "$@")
array="${array%%;*}"
summary=$(sbatch --parsable --mem=2G --cpus-per-task=1 --dependency="afterany:$array" \
  scripts/sbatch/compare_skeleton_geometry.sh summarize "$@")
printf 'Prepare/test: %s\nComparison array: %s\nSummary: %s\n' "$prepare" "$array" "${summary%%;*}"
