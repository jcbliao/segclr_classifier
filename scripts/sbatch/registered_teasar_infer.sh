#!/bin/bash
#SBATCH --job-name=registered_teasar_infer
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -euo pipefail
REPO=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
if [ "${MODE:-infer}" = pilot ]; then
  exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/infer_named_teasar.py --pilot
fi
/home/jcbliao/.conda/envs/segclr/bin/python -u scripts/registered_teasar_pipeline.py infer \
  --phase "$PHASE" --partition "$GPU_POOL" --rank "$SLURM_ARRAY_TASK_ID" --world "$GPU_TASKS"
