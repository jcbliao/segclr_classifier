#!/bin/bash
# Verifies a bounding_box-restricted mesh.get() call actually returns a small
# LOCAL patch, not a whole-neuron mesh. One CAVE call, one CloudVolume mesh
# fetch of a small bbox -- lightweight, but needs network/CAVE auth so not
# mit_quicktest's usual "no network" assumption; still bounded and quick.
#SBATCH --job-name=check_local_mesh_patch
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

export CAVE_TOKEN=$(jq -r .token ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json)

"$PY" -u scripts/check_local_mesh_patch.py
