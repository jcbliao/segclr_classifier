#!/bin/bash
# Submit one two-part augmentation set, then arrange the next submission only
# after both partitions finish. This respects mit_normal_gpu's two-job submit cap.
set -euo pipefail
cd "$(dirname "$0")/.."

index=${1:?sequence index required}
sets=(gray:0 flip:0 structured_low:0 gray:1 flip:1 structured_low:1)
if (( index < 0 || index >= ${#sets[@]} )); then
  exit 0
fi
item=${sets[$index]}
policy=${item%:*}
sample=${item#*:}

preempt_id=$(POLICY="$policy" SAMPLE="$sample" NUM_TASKS=18 sbatch --parsable \
  --array='0-15%16' --partition=mit_preemptable --account=mit_general --qos=normal \
  --job-name="aug_${policy}_${sample}_pre" \
  scripts/sbatch/generate_cave_embedding_augmentations.sh)
normal_id=$(POLICY="$policy" SAMPLE="$sample" NUM_TASKS=18 sbatch --parsable \
  --array='16-17%2' --partition=mit_normal_gpu --account=mit_amf_standard_gpu \
  --qos=mit_amf_standard_gpu --time=06:00:00 \
  --job-name="aug_${policy}_${sample}_norm" \
  scripts/sbatch/generate_cave_embedding_augmentations.sh)
preempt_id=${preempt_id%%;*}
normal_id=${normal_id%%;*}
printf '%s\t%s\t%s\t%s\n' "$policy" "$sample" "$preempt_id" "$normal_id" \
  >> logs/cave_aug2_continuation.tsv

next=$((index + 1))
if (( next < ${#sets[@]} )); then
  sbatch --parsable --dependency="afterok:${preempt_id}:${normal_id}" \
    --partition=mit_normal --account=mit_amf_standard_cpu --qos=mit_amf_standard_cpu \
    --cpus-per-task=1 --mem=1G --time=00:05:00 \
    --job-name=cave_aug_submit \
    --output="logs/cave_aug_submit_%j.out" --error="logs/cave_aug_submit_%j.err" \
    scripts/continue_cave_embedding_augmentations.sh "$next"
fi
