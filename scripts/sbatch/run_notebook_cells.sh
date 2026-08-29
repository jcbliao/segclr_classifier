#!/bin/bash
# Execute every code cell of a notebook in order, headless. Unlike
# check_notebook_syntax.sh (parse only) this actually runs the cells against
# the data on disk, which is what catches a column name that no longer exists
# or a pivot that collapses -- the failures a parse cannot see.
#
# Figures go to Agg and are discarded; `display` is stubbed, so no Jupyter
# kernel or browser is involved. NOTEBOOK selects the file, relative to the
# repo root; the cells run with the notebook's own directory as cwd, since
# notebooks here address the repo as REPO_ROOT = Path("..").
#SBATCH --job-name=run_notebook_cells
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

NOTEBOOK="${NOTEBOOK:-analysis/training_curves.ipynb}"

MPLBACKEND=Agg "$PY" -u - "$NOTEBOOK" <<'EOF'
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(sys.argv[1]).resolve()
nb = json.loads(path.read_text())
os.chdir(path.parent)
print(f"{path}  (cwd {Path.cwd()})")

# The notebook calls display() and plt.show(); neither exists usefully here.
namespace = {"__name__": "__main__", "display": lambda *a, **k: None}
plt.show = lambda *a, **k: plt.close("all")

failures = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = cell["source"]
    src = src if isinstance(src, str) else "".join(src)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), namespace)
        print(f"cell {i}: OK")
    except Exception:
        failures += 1
        print(f"cell {i}: FAILED")
        traceback.print_exc()
    plt.close("all")

print(f"\n{failures} cell(s) failed to execute")
raise SystemExit(1 if failures else 0)
EOF
