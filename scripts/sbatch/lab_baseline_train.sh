#!/bin/bash
# Trains the lab's actual DeepResNet baseline (segCLR_cell_classification)
# through their own train.py, with only the split logic swapped for ours
# (scripts/lab_baseline_train.py). GPU job per project policy.
#SBATCH --job-name=lab_baseline_train
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segCLR_cell_classification/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier/segCLR_cell_classification

"$PY" -u ../scripts/lab_baseline_train.py ../configs/lab_baseline_resnet.yaml
