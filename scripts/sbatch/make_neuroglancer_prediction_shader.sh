#!/bin/bash
# Generate one paste-ready Neuroglancer layer JSON on a short CPU batch job.
# Usage:
#   MODEL=mpnn_L2_position_lpe_resnet4x128_n20 MODE=rainbow \
#     sbatch scripts/sbatch/make_neuroglancer_prediction_shader.sh
#   MODEL=... MODE=correctness SEGMENT=864691... OUTPUT=/path/config.json \
#     sbatch scripts/sbatch/make_neuroglancer_prediction_shader.sh
#SBATCH --job-name=ng_shader
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
PY="$REPO/segclr_db/.venv/bin/python"
: "${MODEL:?set MODEL to an exact run name or unique substring}"
MODE="${MODE:-rainbow}"
OUTPUT="${OUTPUT:-$REPO/results/all_windows/neuroglancer_shader_config.json}"
segment_args=()
if [[ -n "${SEGMENT:-}" ]]; then
  segment_args=(--segment "$SEGMENT")
fi
cd "$REPO"
exec "$PY" -u scripts/make_neuroglancer_prediction_shader.py \
  "$MODEL" --mode "$MODE" --output "$OUTPUT" "${segment_args[@]}"
