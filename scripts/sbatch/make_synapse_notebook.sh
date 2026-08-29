#!/bin/bash
#SBATCH --job-name=nb_synapses
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
PY="$REPO/segclr_db/.venv/bin/python"

PYTHONPATH="$REPO" "$PY" -u scripts/make_synapse_notebook.py
# Execute in place so the committed notebook carries its figures and tables.
PYTHONPATH="$REPO" "$PY" -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3 \
  analysis/synapse_inventory.ipynb
echo "done"
