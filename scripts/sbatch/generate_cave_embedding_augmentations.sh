#!/bin/bash
#SBATCH --job-name=cave_aug2
#SBATCH --partition=mit_preemptable,mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/home/jcbliao/rotation/segclr/gnn_classifier${PYTHONPATH:+:${PYTHONPATH}}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/jcbliao/.conda/envs/segclr/bin/python -u \
  scripts/generate_cave_embedding_augmentations.py \
  --task-id "${SLURM_ARRAY_TASK_ID}" --num-tasks "${NUM_TASKS:-64}" \
  --policy "${POLICY:?set POLICY}" --sample "${SAMPLE:?set SAMPLE}" "$@"
