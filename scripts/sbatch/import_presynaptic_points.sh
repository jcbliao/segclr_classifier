#!/bin/bash
#SBATCH --job-name=pre_segclr_import
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
segclr_db/.venv/bin/python -u scripts/validate_postsynaptic_embeddings.py --output /orcd/scratch/orcd/013/jcbliao/presynaptic_point_embeddings
exec segclr_db/.venv/bin/python -u scripts/import_postsynaptic_embeddings.py --input /orcd/scratch/orcd/013/jcbliao/presynaptic_point_embeddings --point-set-name presynaptic_sites --report analysis/neuron_presynaptic_points/import_report.json --write
