#!/bin/bash
# Installs reportlab into segclr_db/.venv, used by scripts/make_code_pdf.py to
# typeset the source tree into a single readable PDF. Pure-Python PDF writer;
# pygments (already installed) does the highlighting. Install via sbatch per
# project policy.
#SBATCH --job-name=install_reportlab
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

UV=~/.local/bin/uv
VENV=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv
cd /home/jcbliao/rotation/segclr/gnn_classifier/segclr_db

# --python is not optional here: sbatch inherits VIRTUAL_ENV from the submitting
# shell, and uv will happily install into that one instead (it picked
# segCLR_cell_classification/.venv on the first attempt).
echo "=== undoing the stray install, if any ==="
"$UV" pip uninstall --python \
  /home/jcbliao/rotation/segclr/gnn_classifier/segCLR_cell_classification/.venv/bin/python \
  reportlab || true

echo "=== installing reportlab ==="
"$UV" pip install --python "$VENV/bin/python" reportlab

echo "=== confirming ==="
"$VENV"/bin/python -c "import reportlab, pygments; print(reportlab.Version, pygments.__version__)"
