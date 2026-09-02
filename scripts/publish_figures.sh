#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python scripts/export_interactive_figures.py

if [[ "${1:-}" == "--push" ]]; then
  git add docs/figures
  if ! git diff --cached --quiet -- docs/figures; then
    git commit -m "Update interactive analysis figures" -- docs/figures
  fi
  git push origin HEAD
  echo "Uploaded figures. GitHub Pages will deploy them from the pushed commit."
else
  echo "Built docs/figures/index.html locally. Run '$0 --push' to commit and upload them."
fi
