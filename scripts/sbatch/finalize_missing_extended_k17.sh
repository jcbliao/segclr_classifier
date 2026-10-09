#!/bin/bash
#SBATCH --job-name=extended_k17_finalize
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
segclr_db/.venv/bin/python -u data/build_missing_extended_k17.py finalize
segclr_db/.venv/bin/python -u scripts/build_casey_confidence_sweep.py
segclr_db/.venv/bin/python -u scripts/build_casey_stratified_folds.py
segclr_db/.venv/bin/python - <<'PY'
import csv
from pathlib import Path
root = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17')
for conf in ('0', '0.3', '0.5', '0.7', '0.9'):
    rows = list(csv.DictReader((root / f'conf{conf}/fold0/cohort.csv').open()))
    tc = [row for row in rows if row['casey_fine_type'] == 'thalamocortical']
    assert len(tc) == 30, (conf, len(tc))
    assert sorted(sum(row['held_out_fold'] == str(i) for row in tc) for i in range(5)) == [6] * 5
print('Verified all 30 thalamocortical cells and six per held-out fold at every confidence')
PY
