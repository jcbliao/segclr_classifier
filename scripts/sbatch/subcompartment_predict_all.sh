#!/bin/bash
#SBATCH --job-name=subcomp_geo10_predict
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --array=0-15%4
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/home/jcbliao/.conda/envs/segclr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/subcompartment_predict_all.py "${MODE:-infer}" --rank "${SLURM_ARRAY_TASK_ID:-0}"
