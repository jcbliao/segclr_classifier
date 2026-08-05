#!/bin/bash
# cloudvolume is needed by caveclient's SkeletonClient.generate_bulk_skeletons_async
# (segclr_db.cave.CAVESkeletonSource._request_generation calls it) but isn't
# declared by segclr_db's `cave` extra (pyproject.toml only lists caveclient)
# and isn't installed -- discovered via scripts/check_stuck_chunk.py: an
# explicit generation request failed with
# "ImportError: Could not import cloudvolume". That call is wrapped in a
# broad try/except that only logs at debug level, so build_dataset.py's
# skeleton-generation requests were failing completely silently, leaving the
# readiness-poll loop waiting forever for skeletons that were never actually
# queued.
#SBATCH --job-name=install_cloudvolume
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

echo "=== cloud-volume ==="
"$UV" pip install cloud-volume

echo "=== verify ==="
.venv/bin/python -c "import cloudvolume; print('cloudvolume', cloudvolume.__version__)"
