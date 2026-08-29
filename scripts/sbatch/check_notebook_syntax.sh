#!/bin/bash
# Syntax-check every code cell of a notebook. Parse only -- nothing is
# executed, so no data, GPU or matplotlib backend is involved.
# NOTEBOOK selects the file, relative to the repo root.
#SBATCH --job-name=check_notebook_syntax
#SBATCH --partition=mit_quicktest
#SBATCH --account=mit_general
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.out
#SBATCH --error=/home/jcbliao/rotation/segclr/gnn_classifier/logs/%x_%j.err

set -euo pipefail

PY=/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
cd /home/jcbliao/rotation/segclr/gnn_classifier

NOTEBOOK="${NOTEBOOK:-analysis/training_curves.ipynb}"

"$PY" -u - "$NOTEBOOK" <<'EOF'
import ast
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
nb = json.loads(path.read_text())
print(path)
failures = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = cell["source"]
    src = src if isinstance(src, str) else "".join(src)
    try:
        ast.parse(src)
        print(f"cell {i}: OK ({len(src.splitlines())} lines)")
    except SyntaxError as exc:
        failures += 1
        print(f"cell {i}: SYNTAX ERROR line {exc.lineno}: {exc.msg}")
        print(f"    {(exc.text or '').rstrip()}")

print(f"\n{failures} cell(s) failed to parse")
raise SystemExit(1 if failures else 0)
EOF
