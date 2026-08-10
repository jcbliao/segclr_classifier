#!/bin/bash
# Installs matplotlib into segclr_db/.venv for the analysis notebooks
# (analysis/*.ipynb). Package install via sbatch per project policy.
#SBATCH --job-name=install_matplotlib
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

UV=~/.local/bin/uv
cd /home/jcbliao/rotation/segclr/gnn_classifier/segclr_db

echo "=== installing matplotlib ==="
"$UV" pip install matplotlib

echo "=== confirming ==="
.venv/bin/python -c "import matplotlib; print(matplotlib.__version__)"
