#!/bin/bash
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
verification=24552134
windows=$(sbatch --parsable --array=0-7%4 --dependency="afterok:${verification}" scripts/sbatch/build_cave_v1dd_addition.sh windows)
finalize=$(sbatch --parsable --dependency="afterok:${windows}" scripts/sbatch/build_cave_v1dd_addition.sh finalize)
mlp=$(sbatch --parsable --partition=mit_preemptable --account=mit_general --qos=normal --dependency="afterok:${finalize}" --array=0,1,3,4%2 scripts/sbatch/train_cave_v1dd_addition.sh)
gt=$(sbatch --parsable --partition=mit_normal_gpu --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu --dependency="afterok:${finalize}" --array=2,5%2 scripts/sbatch/train_cave_v1dd_addition.sh)
summary=$(sbatch --parsable --dependency="afterok:${mlp}:${gt}" scripts/sbatch/summarize_cave_v1dd_addition.sh)
record=results/presynaptic/cave_v1dd_addition/recovery_$(date +%Y%m%d_%H%M%S).tsv
printf 'stage\tjob_id\nverification\t%s\nwindows\t%s\nfinalize\t%s\nmean_mlp\t%s\ngt\t%s\nsummary\t%s\n' "$verification" "$windows" "$finalize" "$mlp" "$gt" "$summary" > "$record"
cat "$record"
