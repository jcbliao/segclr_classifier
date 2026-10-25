#!/bin/bash
#SBATCH --job-name=paired_axon_mlp
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%4
#SBATCH --requeue
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
rank=${SLURM_ARRAY_TASK_ID:?}
fold=$((rank / 2))
condition=axon_only
if (( rank % 2 )); then condition=axon_dendrite; fi
exec segclr_db/.venv/bin/python -u scripts/train_paired_axon_experiment.py --fold "$fold" --condition "$condition" --workers 15
