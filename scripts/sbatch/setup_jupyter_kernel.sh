#!/bin/bash
# Installs ipykernel into segclr_db/.venv and registers it as a Jupyter
# kernel, so segclr_db/examples/tutorial.ipynb (and any other notebook) can
# use this project's actual environment. Package install + kernel
# registration both go through sbatch per project policy (all compute,
# not just training code, runs via sbatch -- see CLAUDE.md hard constraint 4).
#SBATCH --job-name=setup_jupyter_kernel
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

UV=~/.local/bin/uv
VENV=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv
PY="$VENV/bin/python"

cd /home/jcbliao/rotation/segclr/gnn_classifier/segclr_db

echo "=== installing ipykernel ==="
"$UV" pip install ipykernel

echo "=== registering kernel ==="
"$PY" -m ipykernel install --user --name segclr_db --display-name "segclr_db (.venv)"

echo "=== installed kernels ==="
"$PY" -m jupyter kernelspec list
