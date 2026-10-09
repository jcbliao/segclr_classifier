#!/bin/bash
#SBATCH --job-name=test_dense_presyn
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/test_dense_presyn_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/home/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
segclr_db/.venv/bin/python -m pytest scripts/test_dense_presynaptic.py -q
segclr_db/.venv/bin/python scripts/train_gnn.py --help > /dev/null
