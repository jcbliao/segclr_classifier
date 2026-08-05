#!/bin/bash
# Installs the GNN-side dependencies (torch, torch-geometric, scikit-learn)
# into the shared segclr_db venv. Run as a batch job rather than interactively
# -- these are multi-GB downloads plus a compiled-extension build step for
# torch-geometric, better suited to a job allocation than the login/dev shell.
#SBATCH --job-name=setup_env
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

export RAYON_NUM_THREADS=1
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_INSTALLS=1

cd /home/jcbliao/rotation/segclr/gnn_classifier/segclr_db

UV=~/.local/bin/uv

echo "=== torch (cu124) ==="
"$UV" pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== torch-geometric ==="
"$UV" pip install torch-geometric

echo "=== scikit-learn ==="
"$UV" pip install scikit-learn

echo "=== versions ==="
.venv/bin/python -c "
import torch, torch_geometric, sklearn
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('torch_geometric', torch_geometric.__version__)
print('sklearn', sklearn.__version__)
"
