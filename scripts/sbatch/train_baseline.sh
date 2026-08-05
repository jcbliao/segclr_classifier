#!/bin/bash
# Trains the geodesic-mean baseline classifier. Project policy: all
# training/eval/inference runs on GPU nodes (mit_normal_gpu), even though this
# particular model is small enough that CPU would technically be fine.
#SBATCH --job-name=train_baseline
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" scripts/train_baseline.py --depth "${DEPTH:-2}"
