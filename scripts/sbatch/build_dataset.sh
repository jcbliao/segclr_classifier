#!/bin/bash
# Builds the local dataset (labels + CAVE skeletons + public embeddings ->
# data/graph_cache/*.pt + data/manifest.json). CPU-only, no segclr_db store.
# mit_normal (not mit_quicktest): CAVE fetching + ~400 GCS zip downloads may
# not fit in 15 min even though most skeletons already exist. Resumable --
# safe to re-submit if it times out or is interrupted.
#SBATCH --job-name=build_dataset
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

export CAVE_TOKEN=$(jq -r .token ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json)

"$PY" data/build_dataset.py --workers "${WORKERS:-12}"
