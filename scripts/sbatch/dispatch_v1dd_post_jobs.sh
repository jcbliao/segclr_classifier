#!/bin/bash
#SBATCH --job-name=v1dd_post_dispatch
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
status=0
segclr_db/.venv/bin/python -u scripts/dispatch_v1dd_post_jobs.py || status=$?
if [[ "$status" -eq 75 ]]; then scontrol requeue "${SLURM_JOB_ID:?}"; exit 0; fi
exit "$status"
