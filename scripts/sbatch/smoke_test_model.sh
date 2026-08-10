#!/bin/bash
# Import + forward-pass smoke test for gnn/* on synthetic data. Project
# policy: all training/eval/inference runs on GPU nodes.
#
# Stays at 2 CPUs, unlike train_gnn.sh's 32: this runs four synthetic
# 1-20 node graphs with no DataLoader at all, so extra cores would sit
# idle on a shared node.
#SBATCH --job-name=smoke_test_model
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/smoke_test_model.py
