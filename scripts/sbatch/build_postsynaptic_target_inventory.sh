#!/bin/bash
#SBATCH --job-name=postsyn_inventory
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=segclr_db/.venv/bin/python
if [[ "${1:-}" == volumes ]]; then
    "$PY" -u data/build_partner_volumes.py --post analysis/postsynaptic_targets_conf0.7/missing_volume_roots.parquet --output analysis/postsynaptic_targets_conf0.7/volume_cache --request-workers 1 --sleep 1.0 --retry-errors
else
    "$PY" -u scripts/build_postsynaptic_target_inventory.py --stage "${1:-skeletons}"
fi
