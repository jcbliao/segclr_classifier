#!/bin/bash
# Rebuild the feature summaries, then check both analysis notebooks headlessly.
# Usage: sbatch scripts/sbatch/smoke_test_analysis_notebooks.sh
# CPU-only: no inference and no feature join. The summarize pass reduces the
# multi-million-row feature caches already on disk, so it wants memory rather
# than a GPU; the heavy build path is build_feature_prediction_cache.sh.
#SBATCH --job-name=analysis_smoke
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail

REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
cd "$REPO"
mkdir -p logs

# Every cached run, rebuilt: a summary written before the hierarchy classes were
# scored has no `neuron` or `excitatory` bins, and the notebook's class dropdown
# is built from what the summaries hold. With no run names, --summarize-only
# defaults to exactly the runs that have a feature cache AND a live results
# directory, so a cache left behind by an architecture since removed is skipped
# instead of raising on its unparseable name.
# SKIP_SUMMARIZE=1 re-runs only the checks, for when the summaries on disk are
# already current -- the reduction is the slow half and rewrites files other
# jobs may be reading.
if [ "${SKIP_SUMMARIZE:-0}" = "1" ]; then
  echo "=== skipping the summarize pass (SKIP_SUMMARIZE=1) ==="
else
  echo "=== summarizing every cached run ==="
  PYTHONPATH="$REPO" "$PY" -u analysis/all_windows/feature_prediction_correlation.py --summarize-only
fi

echo "=== smoke test ==="
PYTHONPATH="$REPO" "$PY" -u scripts/smoke_test_analysis_notebooks.py
