#!/bin/bash
# tensorstore reads the sharded precomputed MICrONS segmentation, which is what
# data/mask_volume.py counts mask voxels in. The segclr inference pipeline at
# ~/projects/segclr uses it from ~/.conda/envs/segclr; this project keeps its own
# uv venv, so it needs its own copy rather than borrowing the conda env.
#SBATCH --job-name=install_tensorstore
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

export RAYON_NUM_THREADS=1
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_INSTALLS=1

cd /home/jcbliao/rotation/segclr/gnn_classifier/segclr_db
UV=~/.local/bin/uv
VENV_PYTHON=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python

# --python is not optional here. Without it uv resolved to
# segCLR_cell_classification/.venv (a different project's clone) and installed there,
# leaving this venv untouched and the verify step failing on a clean install.
echo "=== tensorstore ==="
"$UV" pip install --python "$VENV_PYTHON" tensorstore

# tensorstore exposes no __version__ attribute; the distribution metadata is the source.
echo "=== verify ==="
"$VENV_PYTHON" -c "import tensorstore, importlib.metadata as m; print('tensorstore', m.version('tensorstore'))"

echo "=== verify the segmentation actually opens ==="
"$VENV_PYTHON" - <<'PY'
import tensorstore as ts

vol = ts.open({
    "driver": "neuroglancer_precomputed",
    "kvstore": {"driver": "file",
                "path": "/orcd/compute/sdorkenw/001/collina/minnie_seg_1718_sharded"},
    "scale_index": 0,
    "open": True,
}).result()
spec = vol.spec().to_json()
print("resolution", spec["scale_metadata"]["resolution"])
print("size      ", spec["scale_metadata"]["size"])
print("domain    ", [(vol.domain[i].inclusive_min, vol.domain[i].exclusive_max)
                     for i in range(3)])
PY
