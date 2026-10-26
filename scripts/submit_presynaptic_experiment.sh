#!/bin/bash
# Chain an already-prepared experiment: five GPU folds, certification, parallel
# database export, and an executed notebook. Existing job records prevent repeats.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
state=$(realpath "$1")
smoke=$2
export_test=$3
exec 9>"$state/submission.lock"
flock -n 9
record="$state/workflow_jobs.json"
tag=$(basename "$state")
if [ ! -e "$record" ]; then
  train=$(sbatch --parsable --job-name="$tag" --array=0-4 --dependency="afterok:$smoke" \
    scripts/sbatch/train_presynaptic_radius.sh train --state "$state")
  jq -n --arg state "$state" --arg smoke "$smoke" --arg train "$train" \
    '{state:$state,gpu_preflight_job:$smoke,training_array:$train}' > "$record"
fi
train=$(jq -r '.training_array' "$record")
needs_job() {
  if [ -z "$1" ]; then return 0; fi
  local status
  status=$(sacct -X -n -P -j "$1" --format=State | head -n 1)
  case "$status" in
    CANCELLED*|FAILED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*) return 0 ;;
    '') echo "Cannot resolve existing job $1; refusing duplicate submission" >&2; exit 1 ;;
    *) return 1 ;;
  esac
}
summary=$(jq -r '.summary_job // empty' "$record")
if needs_job "$summary"; then
  summary=$(sbatch --parsable --job-name="${tag}_summary" --partition=mit_normal --account=mit_amf_advanced_cpu --qos=mit_amf_advanced_cpu \
    --dependency="afterok:$train" scripts/sbatch/summarize_presynaptic_radius.sh --state "$state")
fi
jq --arg job "$summary" '. + {summary_job:$job}' "$record" > "$record.tmp"
mv "$record.tmp" "$record"
publish=$(jq -r '.export_job // empty' "$record")
if needs_job "$publish"; then
  publish=$(sbatch --parsable --job-name="${tag}_export" --dependency="afterok:$summary:$export_test" \
    scripts/sbatch/export_presynaptic_radius.sh --state "$state")
fi
jq --arg job "$publish" '. + {export_job:$job}' "$record" > "$record.tmp"
mv "$record.tmp" "$record"
notebook=$(jq -r '.notebook_job // empty' "$record")
if needs_job "$notebook"; then
  notebook=$(sbatch --parsable --job-name="${tag}_notebook" --partition=mit_normal --account=mit_amf_advanced_cpu --qos=mit_amf_advanced_cpu \
    --dependency="afterok:$publish" scripts/sbatch/build_presynaptic_radius_notebook.sh "$state")
fi
workflow_status=submitted
if [ "$(sacct -X -n -P -j "$notebook" --format=State | head -n 1)" = COMPLETED ]; then
  workflow_status=complete
fi
jq --arg job "$notebook" --arg status "$workflow_status" '. + {notebook_job:$job,status:$status}' "$record" > "$record.tmp"
mv "$record.tmp" "$record"
cat "$record"
