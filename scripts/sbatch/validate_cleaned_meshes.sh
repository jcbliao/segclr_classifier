#!/bin/bash
#SBATCH --job-name=mesh_validate_live
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.out
#SBATCH --error=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%A_%a.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec segclr_db/.venv/bin/python -u scripts/finalize_cleaned_mesh_layer.py \
 --output "$OUTPUT" --validation-only --task-id "$SLURM_ARRAY_TASK_ID" \
 --num-tasks 16 --watch-jobs 24963728,24963729,24963730,24964410
