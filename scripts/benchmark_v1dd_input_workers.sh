#!/bin/bash
# Measure the current 32x8 array, then replace it with 64x4.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
old_array=${1:-24488906}
sample_seconds=${V1DD_BENCHMARK_SECONDS:-300}
output=analysis/v1dd_cell_types/worker_benchmark
mkdir -p "$output"
# Fail before waiting or changing jobs if the scheduler is inaccessible.
squeue -h -j "$old_array" > "$output/baseline_queue.txt"
segclr_db/.venv/bin/python scripts/measure_v1dd_input_rate.py \
  --job-id "$old_array" --seconds "$sample_seconds" --output "$output/32_jobs_8_cpus.json"
# Stage the replacement held, so a rejected submission preserves the old array.
new_array=$(sbatch --parsable --hold --array=0-63%64 --cpus-per-task=4 --mem=16G \
  scripts/sbatch/prepare_v1dd_inputs.sh)
new_array=${new_array%%;*}
printf '%s\n' "$new_array" > "$output/replacement_array.txt"
scancel "$old_array"
scontrol release "$new_array"
printf 'Submitted 64x4 CPU array %s; waiting for at least one worker.\n' "$new_array"
while true; do
  running=$(squeue -h -r -j "$new_array" -t RUNNING -o '%i')
  if [[ -n "$running" ]]; then break; fi
  sleep 10
done
sleep 60
segclr_db/.venv/bin/python scripts/measure_v1dd_input_rate.py \
  --job-id "$new_array" --seconds "$sample_seconds" --output "$output/64_jobs_4_cpus.json"
segclr_db/.venv/bin/python - "$output" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
for name in ('32_jobs_8_cpus', '64_jobs_4_cpus'):
    r=json.loads((p/f'{name}.json').read_text())
    print(f"{name}: {r['ready_inputs_per_minute']:.1f} ready inputs/min; "
          f"workers at sample end: {len([x for x in r['queue_after'] if ' RUNNING ' in x])}")
print('This compares successive production intervals; cache coverage and cell workload can differ.')
PY
