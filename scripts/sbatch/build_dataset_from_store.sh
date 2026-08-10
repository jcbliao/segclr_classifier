#!/bin/bash
# Builds the GNN dataset from the real segclr-db store (properly-indexed,
# replaces the deprecated nearest-neighbor pipeline). CPU-only data prep, not
# model training -- but reading Lance tables at this scale needs real memory
# (see scripts/explore_real_store.py's OOM lesson at 8G).
#SBATCH --job-name=build_dataset_from_store
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

# No CAVE_TOKEN needed anymore -- labels come from segclr_db's own
# registered cell_labels table (db.get_labels()), not a direct CAVE query.
"$PY" -u data/build_dataset_from_store.py --workers "${WORKERS:-8}"
