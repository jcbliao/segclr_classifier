#!/bin/bash
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
OUTPUT=/orcd/compute/sdorkenw/001/jcbliao/neuroglancer/microns/cleaned_teasar
export OUTPUT
# Arrays of <=800 tasks fit the common MaxArraySize=1001 configuration.
# No % concurrency cap: SLURM enforces account and partition limits.
for roots in "$OUTPUT"/batches/*.txt; do
  if [[ -f "$OUTPUT/submitted_jobs.tsv" ]] && awk -F '\t' -v roots="$roots" '$1 == roots {found=1} END {exit !found}' "$OUTPUT/submitted_jobs.tsv"; then
    continue
  fi
  count=$(wc -l < "$roots")
  workers=240
  if [[ "$roots" == *normal_* ]]; then
    sched=(--partition=mit_normal --account=mit_amf_advanced_cpu --qos=mit_amf_advanced_cpu)
  else
    sched=(--partition=mit_preemptable --account=mit_general --qos=normal)
    # Two arrays share the mit_general 500-submitted-job limit.
    workers=240
    if [[ "$roots" == *preemptable_01.txt ]]; then workers=208; fi
  fi
  export ROOTS="$roots"
  export WORKERS="$workers"
  job=$(sbatch --parsable "${sched[@]}" --array="0-$((workers-1))" scripts/sbatch/rebuild_cleaned_meshes.sh)
  printf '%s\t%s\n' "$roots" "$job" >> "$OUTPUT/submitted_jobs.tsv"
  printf 'Submitted %s: %s cells (%s)\n' "$job" "$count" "$roots"
done
