#!/bin/bash
#SBATCH --job-name=skeleton_geometry
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=segclr_db/.venv/bin/python
MODE="${1:-run}"
if (( $# > 0 )); then shift; fi
if [[ "$MODE" == test ]]; then
    exec "$PY" -u scripts/test_compare_skeleton_geometry.py -v
fi
if [[ "$MODE" == prepare ]]; then
    "$PY" -u scripts/test_compare_skeleton_geometry.py -v
fi
exec "$PY" -u scripts/compare_skeleton_geometry.py "$MODE" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" --num-tasks "${NUM_TASKS:-1}" \
  --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
