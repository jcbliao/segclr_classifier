#!/bin/bash
#SBATCH --job-name=pre_geo20_notebook
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
state=${1:-analysis/presynaptic/latest100k_geodesic20um}
state=$(realpath --relative-to="$PWD" "$state")
notebook="$state/five_fold_results.ipynb"
segclr_db/.venv/bin/python -u scripts/build_presynaptic_radius_notebook.py --state "$state"
segclr_db/.venv/bin/python -u scripts/verify_notebooks.py "$notebook" --timeout 300
cp "logs/notebook_checks/$SLURM_JOB_ID/$notebook" "$notebook"
cp "logs/notebook_checks/$SLURM_JOB_ID/report.json" "$state/notebook_verification.json"
if [ -f "$state/workflow_jobs.json" ]; then
  jq '.status = "complete"' "$state/workflow_jobs.json" > "$state/workflow_jobs.json.tmp"
  mv "$state/workflow_jobs.json.tmp" "$state/workflow_jobs.json"
fi
