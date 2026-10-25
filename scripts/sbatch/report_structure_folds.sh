#!/bin/bash
#SBATCH --job-name=report_structure
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=node3006
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/report_structure_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/report_structure_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fold=$((SLURM_ARRAY_TASK_ID / 4 + 1))
seed=$((SLURM_ARRAY_TASK_ID % 4))
if [ "$seed" = 3 ]; then
    exec segclr_db/.venv/bin/python -u scripts/analyze_whole_cell_structure.py inspect --fold "$fold"
else
    exec segclr_db/.venv/bin/python -u scripts/analyze_whole_cell_structure.py evaluate --fold "$fold" --seed "$seed"
fi
