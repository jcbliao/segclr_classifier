#!/bin/bash
# Train selected cell-held-out folds; defaults to fold 0 for the pilot.
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
dependency_args=()
if [ -n "${1:-}" ]; then
  dependency_args=(--dependency="afterok:$1")
fi

preempt_tasks=()
gt_tasks=()
folds=${CASEY_FOLDS:-0}
for fold in $folds; do
  if [[ ! "$fold" =~ ^[0-4]$ ]]; then
    echo "Invalid fold: $fold (expected 0 through 4)" >&2
    exit 2
  fi
  for confidence_index in 0 1 2 3 4; do
    task=$((fold * 15 + confidence_index * 3))
    preempt_tasks+=("$task" "$((task + 1))")
    gt_tasks+=("$((task + 2))")
  done
done
preempt_array=$(IFS=,; printf '%s' "${preempt_tasks[*]}")
gt_array=$(IFS=,; printf '%s' "${gt_tasks[*]}")

preemptable=$(sbatch --parsable \
  --partition=mit_preemptable --account=mit_general --qos=normal \
  "${dependency_args[@]}" \
  --array="${preempt_array}%4" \
  scripts/sbatch/train_casey_confidence_k17.sh)
normal_gpu=$(sbatch --parsable \
  "${dependency_args[@]}" \
  --array="${gt_array}%2" \
  scripts/sbatch/train_casey_confidence_k17.sh)
summary=$(sbatch --parsable \
  --dependency="afterok:${preemptable},afterok:${normal_gpu}" \
  --export="ALL,CASEY_FOLDS=${folds// /,}" \
  scripts/sbatch/summarize_casey_confidence_k17.sh)

mkdir -p results/presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17
record="results/presynaptic/casey_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17/submission_$(date +%Y%m%d_%H%M%S).tsv"
printf 'stage\tjob_id\ttasks\tconcurrency\tfolds\npreemptable\t%s\tmean,pointwise_mlp\t4\t%s\nnormal_gpu\t%s\tgraph_transformer\t2\t%s\nsummary\t%s\tselected\t\t%s\n' \
  "$preemptable" "$folds" "$normal_gpu" "$folds" "$summary" "$folds" > "$record"
cat "$record"
