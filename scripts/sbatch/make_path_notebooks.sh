#!/bin/bash
#SBATCH --job-name=nb_paths
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/nb_paths_%j.out
#SBATCH --error=logs/nb_paths_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"; mkdir -p logs
PY="$REPO/segclr_db/.venv/bin/python"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_$SLURM_JOB_ID"; mkdir -p "$NUMBA_CACHE_DIR"
PYTHONPATH="$REPO" "$PY" -u scripts/make_path_notebooks.py
# Execute in place so the committed notebooks carry their figures and tables.
for nb in embedding_path_geodesics soma_restriction; do
  echo "=== executing $nb ==="
  PYTHONPATH="$REPO" "$PY" -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=python3 \
    "analysis/$nb.ipynb"
done
echo "done"
