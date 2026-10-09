#!/bin/bash
# Queue preparation now, and classifier windows/training behind verified inference.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
verification=${1:-24474619}
mkdir -p logs results/presynaptic/cave_v1dd_addition analysis/presynaptic/cave_v1dd_addition
sites=$(sbatch --parsable --array=0-7%4 scripts/sbatch/build_cave_v1dd_addition.sh topology_sites)
windows=$(sbatch --parsable --array=0-7%4 --dependency="afterok:${sites}:${verification}" scripts/sbatch/build_cave_v1dd_addition.sh windows)
finalize=$(sbatch --parsable --dependency="afterok:${windows}" scripts/sbatch/build_cave_v1dd_addition.sh finalize)
preemptable=$(sbatch --parsable --partition=mit_preemptable --account=mit_general --qos=normal \
  --dependency="afterok:${finalize}" --array=0,1,3,4%2 \
  scripts/sbatch/train_cave_v1dd_addition.sh)
normal=$(sbatch --parsable --partition=mit_normal_gpu --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu \
  --dependency="afterok:${finalize}" --array=2,5%2 \
  scripts/sbatch/train_cave_v1dd_addition.sh)
summary=$(sbatch --parsable --dependency="afterok:${preemptable}:${normal}" scripts/sbatch/summarize_cave_v1dd_addition.sh)
record=results/presynaptic/cave_v1dd_addition/submission_$(date +%Y%m%d_%H%M%S).tsv
printf 'stage\tjob_id\tdependency\nverified_inference\t%s\t\ntopology_synapses\t%s\t\nwindows\t%s\tafterok:%s:%s\nfinalize\t%s\tafterok:%s\nmean_mlp\t%s\tafterok:%s\ngt\t%s\tafterok:%s\nsummary\t%s\tafterok:%s:%s\n' \
  "$verification" "$sites" "$windows" "$sites" "$verification" "$finalize" "$windows" \
  "$preemptable" "$finalize" "$normal" "$finalize" "$summary" "$preemptable" "$normal" > "$record"
cat "$record"
