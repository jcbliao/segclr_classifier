#!/bin/bash
# Prerequisites are explicit so rerunning cannot silently replace frozen cohorts.
set -euo pipefail
: "${PRIORITY_PREP_JOB:?}" "${REMAINING_PREP_JOB:?}" "${PILOT_JOB:?}"
REPO=/orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
OUT=/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911
cd "$REPO"
record="$OUT/inference_jobs.tsv"
if [ -s "$record" ]; then
  cat "$record"
  exit 0
fi
printf 'stage\tjob_id\tpartition\tconcurrency\tdependency\n' > "$record"
submit_gpu() {
  local phase="$1" pool="$2" tasks="$3" concurrency="$4" dependency="$5"
  local sched=()
  if [ "$pool" = preemptable ]; then
    sched=(--partition=mit_preemptable --account=mit_general --qos=normal)
  else
    sched=(--partition=mit_normal_gpu --account=mit_amf_standard_gpu --qos=mit_amf_standard_gpu)
  fi
  local job
  job=$(PHASE="$phase" GPU_POOL="$pool" GPU_TASKS="$tasks" sbatch --parsable \
    "${sched[@]}" --array="0-$((tasks-1))%$concurrency" \
    --dependency="$dependency" --job-name="segclr_${phase}_${pool}" \
    --output="$OUT/logs/infer_${phase}_${pool}_%A_%a.out" scripts/sbatch/registered_teasar_infer.sh) || return "$?"
  job="${job%%;*}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${phase}_${pool}" "$job" "$pool" "$concurrency" "$dependency" >> "$record"
  printf '%s' "$job"
}
priority_preempt=$(submit_gpu priority preemptable 96 4 "afterok:$PRIORITY_PREP_JOB:$PILOT_JOB")
priority_normal=$(submit_gpu priority normal 48 2 "afterok:$PRIORITY_PREP_JOB:$PILOT_JOB")
priority_verify=$(MODE=verify PHASE=priority sbatch --parsable --time=00:15:00 \
  --dependency="afterok:$priority_preempt:$priority_normal" --job-name=verify_priority_teasar \
  --output="$OUT/logs/verify_priority_%j.out" scripts/sbatch/registered_teasar_cpu.sh)
printf 'priority_verify\t%s\tcpu\t1\tafterok:%s:%s\n' "$priority_verify" "$priority_preempt" "$priority_normal" >> "$record"
remaining_preempt=$(submit_gpu remaining preemptable 16 4 "afterok:$priority_verify:$REMAINING_PREP_JOB")
remaining_normal=$(submit_gpu remaining normal 8 2 "afterok:$priority_verify:$REMAINING_PREP_JOB")
remaining_verify=$(MODE=verify PHASE=remaining sbatch --parsable --time=00:15:00 \
  --dependency="afterok:$remaining_preempt:$remaining_normal" --job-name=verify_remaining_teasar \
  --output="$OUT/logs/verify_remaining_%j.out" scripts/sbatch/registered_teasar_cpu.sh)
printf 'remaining_verify\t%s\tcpu\t1\tafterok:%s:%s\n' "$remaining_verify" "$remaining_preempt" "$remaining_normal" >> "$record"
cat "$record"
