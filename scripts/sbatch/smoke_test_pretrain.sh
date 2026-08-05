#!/bin/bash
# Smoke test for the real pretrain_gnn.py script against a tiny synthetic
# dataset (see scripts/smoke_test_pretrain.py). Project policy: all
# training/eval/inference runs on GPU nodes.
#SBATCH --job-name=smoke_test_pretrain
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" scripts/smoke_test_pretrain.py
