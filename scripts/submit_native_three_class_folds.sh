#!/bin/bash
# Prepare and verify label mappings before launching all 15 trainings.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
mkdir -p logs
prepare_job=$(sbatch --parsable scripts/sbatch/prepare_native_three_class.sh)
prepare_job=${prepare_job%%;*}
printf 'Preparation job: %s\n' "$prepare_job"
for experiment in native_single_pre_post native_skeletons_pre_post; do
  mkdir -p "analysis/presynaptic/$experiment/three_class"
  jq -n --arg prepare_job "$prepare_job" \
    '{prepare_job: $prepare_job, status: "preparation submitted; training not yet submitted"}' \
    > "analysis/presynaptic/$experiment/three_class/submission.json"
done
training_job=$(sbatch --parsable --dependency="afterok:$prepare_job" scripts/sbatch/train_native_three_class_folds.sh)
training_job=${training_job%%;*}
printf 'Training array: %s (15 runs, folds 0–4)\n' "$training_job"
for experiment in native_single_pre_post native_skeletons_pre_post; do
  jq -n --arg prepare_job "$prepare_job" --arg training_job "$training_job" \
    '{prepare_job: $prepare_job, training_array: $training_job, folds: [0,1,2,3,4],
      classes: ["BasketFam", "nonBasketInhibitory", "Excitatory"],
      task_mapping: "fold = task / 3; task % 3: 0 = single MLP, 1 = k17 MLP, 2 = k17 GT",
      condition: "pre_post", epochs: 30, status: "submitted"}' \
    > "analysis/presynaptic/$experiment/three_class/submission.json"
done
