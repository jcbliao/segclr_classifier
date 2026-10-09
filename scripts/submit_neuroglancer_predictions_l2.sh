#!/bin/bash
# Prepare -> resumable parallel exact-node L2 export -> atomic publish.
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
SHARDS="${SHARDS:-16}"
if ! [[ "$SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARDS must be a positive integer" >&2
  exit 2
fi

prepare=$(sbatch --parsable scripts/sbatch/prepare_neuroglancer_predictions_l2.sh "$@")
prepare_id="${prepare%%;*}"
array=$(sbatch --parsable --dependency="afterok:$prepare_id" \
  --export="ALL,PLAN_KEY=$prepare_id" --array="0-$((SHARDS - 1))" \
  scripts/sbatch/export_neuroglancer_l2_shard.sh)
array_id="${array%%;*}"
finalize=$(sbatch --parsable --dependency="afterok:$array_id" \
  --export="ALL,PLAN_KEY=$prepare_id" \
  scripts/sbatch/finalize_neuroglancer_predictions_l2.sh)

echo "prepare  $prepare_id"
echo "export   $array ($SHARDS shards)"
echo "finalize $finalize"
