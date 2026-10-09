#!/bin/bash
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
mkdir -p logs analysis/presynaptic/cave_skeletons_pre_post/three_class
prepare_job=$(sbatch --parsable scripts/sbatch/prepare_cave_three_class_pre_post.sh)
prepare_job=${prepare_job%%;*}
printf 'Preparation job: %s\n' "$prepare_job"
training_job=$(sbatch --parsable --dependency="afterok:$prepare_job" scripts/sbatch/train_cave_three_class_pre_post_folds.sh)
training_job=${training_job%%;*}
printf 'CAVE training array: %s (two models, five folds)\n' "$training_job"
jq -n --arg preparation "$prepare_job" --arg training "$training_job" \
  '{prepare_job:$preparation, training_array:$training, status:"submitted", folds:[0,1,2,3,4],
    architectures:["pointwise_mlp","graph_transformer"], num_embeddings:10,
    classes:["BasketFam","nonBasketInhibitory","Excitatory"], condition:"pre_post",
    task_mapping:"fold = task / 2; task % 2: 0 = MLP, 1 = GT", epochs:30}' \
  > analysis/presynaptic/cave_skeletons_pre_post/three_class/submission.json
