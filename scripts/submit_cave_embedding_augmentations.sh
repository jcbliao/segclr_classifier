#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

tasks=${NUM_TASKS:-64}
if (( tasks % 2 != 0 )); then
  echo "NUM_TASKS must be even" >&2
  exit 2
fi
half=$((tasks / 2))
dependency=${INITIAL_DEPENDENCY:-}
record="logs/cave_aug2_submission_$(date +%Y%m%d_%H%M%S).tsv"
mkdir -p logs
printf 'policy\tsample\tpartition\tjob_id\n' > "$record"

for item in ${ITEMS:-gray:0 flip:0 structured_low:0 gray:1 flip:1 structured_low:1}; do
  policy=${item%:*}
  sample=${item#*:}
  args=(--parsable)
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:${dependency}")
  fi
  preempt_id=$(POLICY="$policy" SAMPLE="$sample" NUM_TASKS="$tasks" \
    sbatch "${args[@]}" --array="0-$((half-1))%16" \
    --partition=mit_preemptable --account=mit_general --qos=normal \
    --job-name="aug_${policy}_${sample}_pre" \
    scripts/sbatch/generate_cave_embedding_augmentations.sh)
  normal_id=$(POLICY="$policy" SAMPLE="$sample" NUM_TASKS="$tasks" \
    sbatch "${args[@]}" --array="$half-$((tasks-1))%2" \
    --partition=mit_normal_gpu --account=mit_amf_standard_gpu \
    --qos=mit_amf_standard_gpu --time=06:00:00 \
    --job-name="aug_${policy}_${sample}_norm" \
    scripts/sbatch/generate_cave_embedding_augmentations.sh)
  preempt_id=${preempt_id%%;*}
  normal_id=${normal_id%%;*}
  dependency="${preempt_id}:${normal_id}"
  printf '%s\t%s\t%s\t%s\n' "$policy" "$sample" mit_preemptable "$preempt_id" >> "$record"
  printf '%s\t%s\t%s\t%s\n' "$policy" "$sample" mit_normal_gpu "$normal_id" >> "$record"
done

cat "$record"
