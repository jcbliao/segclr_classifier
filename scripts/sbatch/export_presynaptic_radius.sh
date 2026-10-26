#!/bin/bash
#SBATCH --job-name=pre_geo20_export
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${1:-}" = inspect ]; then
  exec segclr_db/.venv/bin/python -u scripts/inspect_radius_export.py
fi
exec segclr_db/.venv/bin/python -u scripts/export_presynaptic_radius_db.py "$@"
