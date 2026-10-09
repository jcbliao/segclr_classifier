#!/bin/bash
#SBATCH --job-name=repo_catalog
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/repo_catalog_%j.out
#SBATCH --error=logs/repo_catalog_%j.err
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$REPO"
PY="$REPO/segclr_db/.venv/bin/python"
"$PY" -u scripts/build_model_catalog.py
"$PY" -u scripts/check_repository.py
"$PY" -m unittest scripts.test_training_config scripts.test_model_catalog
"$PY" -u scripts/train_gnn.py --config configs/graph_transformer/fixed_node.json --help > "logs/training_cli_help_${SLURM_JOB_ID}.txt"
