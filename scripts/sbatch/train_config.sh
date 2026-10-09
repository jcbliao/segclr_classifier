#!/bin/bash
#SBATCH --job-name=gnn_config
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/gnn_config_%j.out
#SBATCH --error=logs/gnn_config_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [ "$#" -lt 1 ]; then
  echo 'Usage: sbatch scripts/sbatch/train_config.sh CONFIG --results-dir NEW_DIRECTORY [trainer flags]' >&2
  exit 2
fi
CONFIG="$1"
shift
# Keep historical runs safe by requiring an explicit output namespace.
HAS_RESULTS=0
for ARG in "$@"; do
  case "$ARG" in --results-dir|--results-dir=*) HAS_RESULTS=1 ;; esac
done
if [ "$HAS_RESULTS" -ne 1 ]; then
  echo 'Supply --results-dir results/reproductions/EXPERIMENT for the new run.' >&2
  exit 2
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec segclr_db/.venv/bin/python -u scripts/train_gnn.py --config "$CONFIG" "$@"
