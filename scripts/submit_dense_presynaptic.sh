#!/bin/bash
# Run after inference, or set INFERENCE_JOB_IDS to colon-separated SLURM IDs.
# Preparation must already exist; run MODE=prepare with the batch wrapper once.
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
build_out=${OUT:-/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/dense_teasar_multiscale_v2}
test -f "$build_out/manifest.json"
topology_job=${TOPOLOGY_JOB_ID:-}
if [ -z "$topology_job" ]; then
  topology_job=$(MODE=topology NUM_TASKS=32 OUT="$build_out" sbatch --parsable --array=0-31%16 scripts/sbatch/build_dense_presynaptic.sh)
fi
sites_job=$(MODE=sites NUM_TASKS=32 OUT="$build_out" sbatch --parsable --array=0-31%4 scripts/sbatch/build_dense_presynaptic.sh)
dependencies="afterok:${topology_job}:${sites_job}"
if [ -n "${INFERENCE_JOB_IDS:-}" ]; then dependencies="${dependencies}:${INFERENCE_JOB_IDS}"; fi
windows_job=$(MODE=windows NUM_TASKS=32 OUT="$build_out" sbatch --parsable --array=0-31%16 --dependency="$dependencies" scripts/sbatch/build_dense_presynaptic.sh)
final_job=$(MODE=finalize OUT="$build_out" sbatch --parsable --dependency="afterok:${windows_job}" scripts/sbatch/build_dense_presynaptic.sh)
echo "Topology: $topology_job; presynaptic sites: $sites_job; windows: $windows_job; finalize: $final_job"
