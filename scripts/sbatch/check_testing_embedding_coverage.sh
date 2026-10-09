#!/bin/bash
#SBATCH --job-name=test_emb_coverage
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/test_emb_coverage_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/home/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$REPO/segclr_db/.venv/bin/python" -u "$REPO/scripts/check_testing_embedding_coverage.py"
