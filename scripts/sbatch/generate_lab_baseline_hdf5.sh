#!/bin/bash
# Builds the HDF5 the lab's CellTypingDataset expects, from our real data.
# Needs h5py in OUR venv (segclr_db/.venv) since this script imports our own
# baseline/data modules -- installing it here rather than assuming it's
# already present.
#SBATCH --job-name=generate_lab_baseline_hdf5
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

export RAYON_NUM_THREADS=1
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_INSTALLS=1

cd /home/jcbliao/rotation/segclr/gnn_classifier/segclr_db
~/.local/bin/uv pip install h5py

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/generate_lab_baseline_hdf5.py
