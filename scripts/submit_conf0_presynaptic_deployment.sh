#!/bin/bash
# Train the four conf0 counterparts, deploy saved inputs, then import to segclr_db.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
state=analysis/db_deployment/presynaptic/conf0
mkdir -p "$state" logs
if [[ -e "$state/jobs.json" ]]; then
  echo "Submission already recorded in $state/jobs.json; inspect it before resubmitting." >&2
  exit 1
fi
out=/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_conf0_v1
if [[ $# -gt 0 ]]; then
  preparation=$1
else
  preparation=$(sbatch --parsable scripts/sbatch/conf0_presynaptic_cpu.sh prepare)
  preparation=${preparation%%;*}
fi
training=$(sbatch --parsable --array=0-19%4 --dependency="afterok:$preparation" scripts/sbatch/conf0_presynaptic_gpu.sh train)
training=${training%%;*}
plan=$(sbatch --parsable --job-name=conf0_inference_plan --dependency="afterok:$training" scripts/sbatch/conf0_presynaptic_cpu.sh prepare-inference)
plan=${plan%%;*}
pilot=$(sbatch --parsable --job-name=conf0_inference_parity --time=00:30:00 --cpus-per-task=4 --mem=16G --dependency="afterok:$plan" scripts/sbatch/conf0_presynaptic_gpu.sh pilot)
pilot=${pilot%%;*}
predictions=$(sbatch --parsable --job-name=conf0_fragment_predictions --exclude=node3006 --array=0-31%4 --dependency="afterok:$pilot" scripts/sbatch/predict_fragment_cell_types.sh infer --output "$out")
predictions=${predictions%%;*}
audit=$(sbatch --parsable --job-name=conf0_prediction_audit --dependency="afterok:$predictions" scripts/sbatch/conf0_presynaptic_cpu.sh finish)
audit=${audit%%;*}
preflight=$(sbatch --parsable --job-name=conf0_import_preflight --export="ALL,SEGCLR_IMPORT_SOURCE=$out,SEGCLR_IMPORT_STATE=logs/presynaptic_import_conf0" --dependency="afterok:$audit" scripts/sbatch/import_presynaptic_cell_types.sh preflight)
preflight=${preflight%%;*}
import_job=$(sbatch --parsable --job-name=conf0_import --export="ALL,SEGCLR_IMPORT_SOURCE=$out,SEGCLR_IMPORT_STATE=logs/presynaptic_import_conf0" --dependency="afterok:$preflight" scripts/sbatch/import_presynaptic_cell_types.sh import)
import_job=${import_job%%;*}
jq -n --arg preparation "$preparation" --arg training "$training" --arg plan "$plan" --arg pilot "$pilot" --arg predictions "$predictions" --arg audit "$audit" --arg preflight "$preflight" --arg import_job "$import_job" --arg output "$out" \
  '{status:"submitted",label_confidence_min:0,configurations:4,folds:5,training_array:$training,preparation:$preparation,inference_plan:$plan,inference_parity:$pilot,prediction_array:$predictions,prediction_audit:$audit,import_preflight:$preflight,import_job:$import_job,output:$output}' > "$state/jobs.json"
cat "$state/jobs.json"
