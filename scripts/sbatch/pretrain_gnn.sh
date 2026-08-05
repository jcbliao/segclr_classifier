#!/bin/bash
# Masked-autoencoder pretraining of the GNN encoder. GPU job -- cells run up
# to ~20k nodes (CLAUDE.md p99), and message passing over that many nodes for
# ~400 cells x many epochs is squarely GPU territory.
#SBATCH --job-name=pretrain_gnn
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" scripts/pretrain_gnn.py --replace-strategy "${REPLACE_STRATEGY:-random}" --epochs "${EPOCHS:-100}"
