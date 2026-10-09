#!/bin/bash
#SBATCH --job-name=ingest_teasar111
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/ingest_teasar111_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
V5=/home/jcbliao/rotation/segclr/segclr_db_named_v5
export PYTHONPATH="$V5/_pkgroot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$V5"
"$REPO/segclr_db/.venv/bin/python" -m pytest tests/test_named_skeletons.py tests/test_skeletons.py -q
if [ "${1:-}" = '--test-only' ]; then exit 0; fi
"$REPO/segclr_db/.venv/bin/python" -u "$REPO/scripts/ingest_testing_teasar.py" "$@"
