#!/bin/bash
#SBATCH --job-name=build_synapses
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err

# CAVE queries: network-bound, not CPU-bound, so 2 cores and modest memory.
# mit_normal rather than quicktest because the full sweep is ~470 requests per
# rank and cannot be assumed to fit in 15 minutes.
#
#   sbatch scripts/sbatch/build_synapses.sh                       # pilot/solo
#   sbatch --array=0-3%4 scripts/sbatch/build_synapses.sh         # the real run
#   MERGE=1 sbatch scripts/sbatch/build_synapses.sh               # fuse the shards
#
# Array width is kept small on purpose: CAVE's materialization service is shared
# with other labs, and each rank already sleeps between calls.

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

export CAVE_TOKEN=$(jq -r .token ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json)

ARGS="${ARGS:-}"
if [ "${MERGE:-0}" = "1" ]; then
  ARGS="--merge $ARGS"
fi

"$PY" -u data/build_synapses.py $ARGS
