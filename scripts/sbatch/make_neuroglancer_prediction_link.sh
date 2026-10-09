#!/bin/bash
# Generate a compact CAVE/Spelunker link for all held-out neurons of one type.
# MODEL may be an exact run or unique substring. TYPE is one of the eight
# active hierarchy classes; MODE is rainbow (default), correctness, or both.
#
#   MODEL=mpnn_L2_position_lpe_resnet4x128_n20 TYPE=pyramidal MODE=rainbow \
#     sbatch scripts/sbatch/make_neuroglancer_prediction_link.sh
#SBATCH --job-name=ng_link
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
: "${MODEL:?set MODEL to an exact run name or unique substring}"
: "${TYPE:?set TYPE to one of the eight active class names}"
MODE="${MODE:-rainbow}"
OUTPUT="${OUTPUT:-$REPO/results/all_windows/neuroglancer_prediction_link.txt}"
cd "$REPO"
exec "$PY" -u scripts/make_neuroglancer_prediction_link.py \
  "$MODEL" --mode "$MODE" --type "$TYPE" --output "$OUTPUT"
