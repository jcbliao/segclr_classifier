#!/bin/bash
# Train selected cell-held-out folds; defaults to fold 0.
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
dependency_args=()
if [ -n "${1:-}" ]; then
  dependency_args=(--dependency="afterok:$1")
fi

preempt_tasks=()
normal_tasks=()
folds=${CASEY_FOLDS:-0}
for fold in $folds; do
  if [[ ! "$fold" =~ ^[0-4]$ ]]; then
    echo "Invalid fold: $fold (expected 0 through 4)" >&2
    exit 2
  fi
  confidences=(0 0.3 0.5 0.7 0.9)
  for confidence_index in 0 1 2 3 4; do
    cohort="/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_confidence_cave/k10/conf${confidences[$confidence_index]}/fold${fold}"
    if [[ ! -f "$cohort/manifest.json" || ! -f "$cohort/metadata.json" || ! -d "$cohort/cells" ]]; then
      echo "Missing prepared CAVE cohort: $cohort. Run scripts/prepare_casey_confidence_cave.py first." >&2
      exit 2
    fi
    task=$((fold * 15 + confidence_index * 3))
    for model_index in 0 1 2; do
      model_task=$((task + model_index))
      if ((model_task % 2 == 0)); then
        preempt_tasks+=("$model_task")
      else
        normal_tasks+=("$model_task")
      fi
    done
  done
done
preempt_array=$(IFS=,; printf '%s' "${preempt_tasks[*]}")
normal_array=$(IFS=,; printf '%s' "${normal_tasks[*]}")

preemptable=$(sbatch --parsable \
  --partition=mit_preemptable --account=mit_general --qos=normal \
  "${dependency_args[@]}" \
  --array="${preempt_array}%4" \
  scripts/sbatch/train_casey_confidence_cave.sh)
normal_gpu=$(sbatch --parsable \
  --partition=mit_normal_gpu --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu \
  "${dependency_args[@]}" \
  --array="${normal_array}%2" \
  scripts/sbatch/train_casey_confidence_cave.sh)
summary=$(sbatch --parsable \
  --dependency="afterok:${preemptable},afterok:${normal_gpu}" \
  --export="ALL,CASEY_FOLDS=${folds// /,}" \
  scripts/sbatch/summarize_casey_confidence_cave.sh)

mkdir -p results/presynaptic/casey_confidence_cave/k10
record="results/presynaptic/casey_confidence_cave/k10/submission_$(date +%Y%m%d_%H%M%S).tsv"
printf 'stage\tjob_id\ttasks\tconcurrency\tfolds\npreemptable\t%s\tmixed_models\t4\t%s\nnormal_gpu\t%s\tmixed_models\t2\t%s\nsummary\t%s\tselected\t\t%s\n' \
  "$preemptable" "$folds" "$normal_gpu" "$folds" "$summary" "$folds" > "$record"
cat "$record"
