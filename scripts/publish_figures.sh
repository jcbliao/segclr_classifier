#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

publish=0
if [[ "${1:-}" == "--push" ]]; then publish=1; fi
job_id="$(sbatch --parsable --export="ALL,PUBLISH_FIGURES=$publish" \
  scripts/sbatch/export_interactive_figures.sh)"
echo "Submitted figure export as Slurm job $job_id."
echo "Logs: logs/export_figures_${job_id}.out and logs/export_figures_${job_id}.err"
if [[ "$publish" == "1" ]]; then
  echo "The job will commit and push docs/figures only after a successful export."
fi
