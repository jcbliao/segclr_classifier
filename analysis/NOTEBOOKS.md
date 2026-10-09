# Running the analysis notebooks

Select **segclr_db (.venv)** in Jupyter or VS Code. It points to
`segclr_db/.venv/bin/python`, which contains the project's scientific Python,
PyTorch/PyG, database, and notebook dependencies. A generic Python kernel can
miss these dependencies even when the notebook code is correct.

Notebooks discover the repository root from their current directory, so they
can run from their own folders. External data lives at the paths declared in
each notebook or helper. Statistics notebooks use persistent caches; a cache
rebuild can take much longer than displaying an existing cache.

Execute all notebooks under `analysis/`, saving execution copies and a JSON
report without replacing source notebook outputs:

```bash
segclr_db/.venv/bin/python scripts/check_notebooks.py
```

Check only setup/import cells, or execute one notebook:

```bash
segclr_db/.venv/bin/python scripts/check_notebooks.py --setup-only
segclr_db/.venv/bin/python scripts/check_notebooks.py analysis/presynaptic/casey_confidence_cave/casey_confidence_comparison.ipynb
```

Reports and copies are written under `logs/notebook_checks/`. `--timeout`
sets the per-cell limit in seconds (default 1800); heavy statistics/cache
rebuilds may need a larger value. Cells retain their normal behavior, including
any explicitly enabled data or figure exports.

Historical notebooks in `20260811/` and `archive/geodesic_radius_20260828/`
use their adjacent archived results. Pass those paths explicitly to check them.
Notebooks in the separate `segCLR_cell_classification` and `segclr_db`
projects have their own data and environment requirements.
