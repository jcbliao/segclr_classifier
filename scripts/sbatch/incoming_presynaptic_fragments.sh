#!/bin/bash
#SBATCH --job-name=presyn_k10_folds
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
stage="${1:-build}"
if [ "$#" -gt 0 ]; then shift; fi
exec /home/jcbliao/.conda/envs/segclr/bin/python -u scripts/incoming_presynaptic_fragments.py \
  "$stage" --rank "${SLURM_ARRAY_TASK_ID:-0}" --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
