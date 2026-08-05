#!/bin/bash
# Sets up segCLR_cell_classification's own venv (separate from segclr_db's --
# matches segclr_db's own pattern of being a vendored dependency with its own
# environment, avoids version conflicts between two independently-maintained
# projects' torch/numpy pins). This repo ships a uv.lock (unlike segclr_db),
# so `uv sync` is the correct install here, not `uv pip install`.
#SBATCH --job-name=setup_lab_baseline_env
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

cd /home/jcbliao/rotation/segclr/gnn_classifier/segCLR_cell_classification
UV=~/.local/bin/uv

"$UV" venv --python 3.11
"$UV" sync

echo "=== versions ==="
.venv/bin/python -c "
import torch, h5py, torchmetrics, sklearn
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('h5py', h5py.__version__)
print('torchmetrics', torchmetrics.__version__)
print('sklearn', sklearn.__version__)
"
