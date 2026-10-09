#!/bin/bash
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier

dependency=()
if [[ -n "${1:-}" ]]; then dependency=(--dependency="afterok:$1"); fi

pre=$(sbatch --parsable --partition=mit_preemptable --account=mit_general --qos=normal \
  "${dependency[@]}" --array=0-1%2 scripts/sbatch/train_cave_embedding_augmentation.sh)
normal=$(sbatch --parsable --partition=mit_normal_gpu --account=mit_amf_standard_gpu \
  --qos=mit_amf_standard_gpu "${dependency[@]}" --array=2-3%2 \
  scripts/sbatch/train_cave_embedding_augmentation.sh)
summary=$(sbatch --parsable --partition=mit_normal --account=mit_general \
  --cpus-per-task=1 --mem=4G --time=00:10:00 \
  --dependency="afterok:${pre},afterok:${normal}" \
  --wrap='cd /home/jcbliao/rotation/segclr/gnn_classifier && segclr_db/.venv/bin/python scripts/summarize_cave_embedding_augmentation.py')
printf 'preemptable=%s (clean,gray)\nnormal_gpu=%s (flip,structured_low)\nsummary=%s\n' \
  "$pre" "$normal" "$summary"
