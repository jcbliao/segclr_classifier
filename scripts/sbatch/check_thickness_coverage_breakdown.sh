#!/bin/bash
# Decomposes unmeasured dendrite-thickness nodes into by-design ineligibility
# (axon/soma/branch point/bad tangent) vs. real ray-cast failure. Recomputes
# eligibility from cached skeletons -- no mesh fetch, no CAVE call. CPU-only.
#SBATCH --job-name=check_thickness_coverage_breakdown
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

"$PY" -u scripts/check_thickness_coverage_breakdown.py ${LIMIT:+--limit "$LIMIT"}
