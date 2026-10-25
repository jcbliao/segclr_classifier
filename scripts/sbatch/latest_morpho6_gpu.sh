#!/bin/bash
#SBATCH --job-name=latest_morpho6_embeddings
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --exclude=node3006
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
source scripts/sbatch/latest_morpho6_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LANCE_CPU_THREADS=2 LANCE_IO_THREADS=2 RAYON_NUM_THREADS=2 MALLOC_ARENA_MAX=2
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/latest_morpho6_training.py "${1:-embed}" --rank "${SLURM_ARRAY_TASK_ID:-0}" --world 32
