#!/bin/bash
#SBATCH --job-name=oracle_route
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs
exec "$PY" -u scripts/eval_oracle_routing.py "$@"
