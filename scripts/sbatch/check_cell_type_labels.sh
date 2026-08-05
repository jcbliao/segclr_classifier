#!/bin/bash
#SBATCH --job-name=check_cell_type_labels
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

export CAVE_TOKEN=$(jq -r .token ~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json)

"$PY" scripts/check_cell_type_labels.py
