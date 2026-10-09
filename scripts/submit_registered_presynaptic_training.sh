#!/bin/bash
# Submit the coarsest native-skeleton experiment after priority verification.
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
verify=${PRIORITY_VERIFY_JOB:-22600108}
mkdir -p logs results/presynaptic/new_skeletons_native
build=$(sbatch --parsable --array=0-31%8 --dependency="afterok:${verify}" scripts/sbatch/build_registered_presynaptic.sh)
finalize=$(sbatch --parsable --time=00:15:00 --dependency="afterok:${build}" scripts/sbatch/build_registered_presynaptic.sh --finalize)
# Split the three architectures 2:1 across preemptable and normal GPUs.
train_preemptable=$(sbatch --parsable --array=0,2 \
  --partition=mit_preemptable --account=mit_general --qos=normal \
  --dependency="afterok:${finalize}" scripts/sbatch/train_presynaptic_native.sh)
train_normal=$(sbatch --parsable --array=1 \
  --partition=mit_normal_gpu --account=mit_amf_standard_gpu --qos=mit_amf_standard_gpu \
  --dependency="afterok:${finalize}" scripts/sbatch/train_presynaptic_native.sh)
record=results/presynaptic/new_skeletons_native/submission_$(date +%Y%m%d_%H%M%S).tsv
printf 'stage\tjob_id\tdependency\npriority_verify\t%s\t\nbuild_scale32\t%s\tafterok:%s\nfinalize_scale32\t%s\tafterok:%s\ntrain_mean_mlp_preemptable\t%s\tafterok:%s\ntrain_gt_normal\t%s\tafterok:%s\n' "$verify" "$build" "$verify" "$finalize" "$build" "$train_preemptable" "$finalize" "$train_normal" "$finalize" > "$record"
cat "$record"
