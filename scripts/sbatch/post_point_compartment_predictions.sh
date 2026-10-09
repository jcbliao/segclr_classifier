#!/bin/bash
#SBATCH --job-name=post_point_compartments
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/post_point_compartment_predictions.py \
  --rank "${SLURM_ARRAY_TASK_ID:-0}" --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
