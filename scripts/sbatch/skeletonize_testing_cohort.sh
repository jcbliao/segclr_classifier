#!/bin/bash
# Partition/account/array limits are supplied by the submission command.
#SBATCH --job-name=test_teasar111
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/test_teasar111_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/test_teasar111_%A_%a.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
SKEL=/home/jcbliao/rotation/skeletonization
: "${OUT:?}" "${ROOTS:?}" "${GLOBAL_TASKS:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SKEL"
PYTHONPATH="$SKEL" /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/skeletonize_segclr.py \
  --roots "$ROOTS" --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$GLOBAL_TASKS" \
  --out "$OUT" --stagger 60
"$REPO/segclr_db/.venv/bin/python" -u "$REPO/scripts/resample_teasar_dense.py" \
  --roots "$ROOTS" --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$GLOBAL_TASKS" --out "$OUT"
