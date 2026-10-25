#!/bin/bash
#SBATCH --job-name=whole_structure_n10
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=node3006
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/whole_structure_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/whole_structure_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$SLURM_ARRAY_TASK_ID" = 3 ]; then
    exec segclr_db/.venv/bin/python -u scripts/analyze_whole_cell_structure.py inspect
else
    exec segclr_db/.venv/bin/python -u scripts/analyze_whole_cell_structure.py evaluate --seed "$SLURM_ARRAY_TASK_ID"
fi
