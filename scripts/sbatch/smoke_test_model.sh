#!/bin/bash
# Import + forward-pass smoke test for gnn/* on synthetic data. Project
# policy: all training/eval/inference runs on GPU nodes.
#SBATCH --job-name=smoke_test_model
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/smoke_test_model.py
