#!/bin/bash
# New deps for dendrite thickness (data/dendrite_thickness.py, ported from
# isebenius/E-I's src/dendrite_thickness.py per explicit user direction):
#   - meshparty: mesh/skeleton I/O (trimesh_io.Mesh/MeshMeta, skeleton.Skeleton)
#   - trimesh: mesh.ray accessor the ray-casting core uses
#   - embreex: pure-wheel embree3 bindings trimesh's ray module auto-detects,
#     so mesh.ray.intersects_first runs the fast embree path instead of
#     falling back to trimesh's pure-python ray_triangle (which would be far
#     too slow at n_rays=64 x n_passes=5 x n_vertices scale across ~2192 cells)
#   - shapely: only used by dendrite_thickness.py's cross_section_radius QC
#     helper (not the core radius_nm estimator), installed anyway since it's
#     a light pure-python-ish package and keeps that helper usable too
# cloudvolume is already installed (scripts/sbatch/install_cloudvolume.sh).
#SBATCH --job-name=install_dendrite_thickness_deps
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

# `uv pip install` resolves the target venv from $VIRTUAL_ENV (if set) BEFORE
# falling back to cwd -- sbatch inherits the submitting shell's environment
# by default, so a $VIRTUAL_ENV left pointing at some OTHER local venv (e.g.
# segCLR_cell_classification/.venv) silently wins over `cd segclr_db`, and
# `uv` installs there instead with no error. Bit us once already this
# session (installed into the wrong venv, had to uninstall + redo). unset
# here AND pass --python explicitly below, so this is correct regardless of
# whatever the submitting shell's environment happens to be.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
VENV_PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO/segclr_db"
UV=~/.local/bin/uv

echo "=== meshparty + trimesh + embreex + shapely ==="
"$UV" pip install --python "$VENV_PY" meshparty trimesh embreex shapely

echo "=== verify ==="
"$VENV_PY" -c "
import meshparty, trimesh, shapely
print('meshparty', meshparty.__version__)
print('trimesh', trimesh.__version__)
print('shapely', shapely.__version__)
import trimesh.ray as tray
print('trimesh ray backends available:', [m for m in dir(tray) if 'embree' in m.lower() or 'pyembree' in m.lower()])
try:
    import embreex
    print('embreex OK', getattr(embreex, '__version__', '?'))
except ImportError as e:
    print('embreex import FAILED:', e)
"
