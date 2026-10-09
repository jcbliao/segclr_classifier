#!/bin/bash
#SBATCH --job-name=existing_teasar111
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/existing_teasar111_%A_%a.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
OUT=/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_existing_teasar_111nm_20260910
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$REPO/segclr_db/.venv/bin/python" -u "$REPO/scripts/resample_teasar_dense.py" \
  --out "$OUT" --roots "$OUT/available_roots.txt" \
  --source-dir /orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons \
  --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks 32
