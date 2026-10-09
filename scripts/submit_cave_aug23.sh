#!/bin/bash
# Submit the combined seven-flip plus sixteen-gray inference after dependencies.
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier

dependency=${1:-}
dep_args=()
if [[ -n "$dependency" ]]; then
  dep_args=(--dependency="afterok:$dependency")
fi
plan=${PLAN:-$PWD/analysis/presynaptic/cave_aug_18_shards.json}

pre=$(PLAN="$plan" sbatch --parsable "${dep_args[@]}" \
  --partition=mit_preemptable --account=mit_general --qos=normal \
  --array='0-11%4' --job-name=cave_aug_remaining_pre \
  scripts/sbatch/generate_all_cave_augmentations.sh)
normal=$(PLAN="$plan" sbatch --parsable "${dep_args[@]}" \
  --partition=mit_normal_gpu --account=mit_amf_standard_gpu \
  --qos=mit_amf_standard_gpu --array='12-17%2' --job-name=cave_aug_remaining_norm \
  scripts/sbatch/generate_all_cave_augmentations.sh)
pre=${pre%%;*}; normal=${normal%%;*}
record="logs/cave_aug_remaining_submission_$(date +%Y%m%d_%H%M%S).tsv"
printf 'stage\tjob_id\ttasks\tconcurrency\tvariants\npreemptable\t%s\t0-11\t4\tgray0-19,flip0-6,structured_low0-3\nnormal_gpu\t%s\t12-17\t2\tgray0-19,flip0-6,structured_low0-3\n' \
  "$pre" "$normal" > "$record"
cat "$record"
