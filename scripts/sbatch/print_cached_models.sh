#!/bin/bash
# What the feature notebook's first cell prints: which runs have a best
# checkpoint, which of those have a summary cache, and what architectures are
# on disk. Cheap and CPU-only, but it imports torch through data.dataset_lcpn,
# so it still goes through sbatch like everything else.
#SBATCH --job-name=cached_models
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
cd "$REPO"
# Heredoc rather than a file under /tmp: /tmp is node-local, so a script
# written on the login node is not there when the job lands on a compute node.
PYTHONPATH="$REPO" "$REPO/segclr_db/.venv/bin/python" -u - <<'PY'
from analysis.all_windows.architecture_comparison import architecture_rank
from analysis.all_windows.feature_prediction_correlation import (
    architecture_of, cached_runs, has_summary, model_of, usable_runs)

runs = usable_runs()
print(f'{len(runs)} runs with a best checkpoint, {sum(map(has_summary, runs))} cached')
print('\nmodels with a cache:',
      *sorted({model_of(r) for r in runs if has_summary(r)}), sep='\n  ')
print('\nmodels still uncached:',
      *sorted({model_of(r) for r in runs if not has_summary(r)}), sep='\n  ')
print('\narchitectures on disk (depth included):',
      *sorted({architecture_of(r) for r in runs}, key=architecture_rank), sep='\n  ')
grid = cached_runs(runs)
print(f'\ncached_runs() grids {len(grid)} runs across '
      f'{len({model_of(r) for r in grid})} model panels')
PY
