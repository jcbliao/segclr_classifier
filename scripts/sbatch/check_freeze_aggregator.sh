#!/bin/bash
# Verifies --freeze-aggregator on both parameterized architectures.
# GPU per CLAUDE.md's rule that model code runs on a GPU node; mit_preemptable
# because quicktest has none. Runs in seconds, so preemption risk is moot.
#SBATCH --job-name=check_freeze_aggregator
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_freeze_aggregator.py
