#!/bin/bash
# Peak GraphTransformer VRAM per (radius, batch size). Requests an L40S
# specifically: the question is whether the GT runs fit on that card, and a
# probe on an H200 would not answer it.
#SBATCH --job-name=check_gt_vram_h200
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_gt_vram.py --radii 40000 --no-embeddings --batch-sizes 512 256 128
