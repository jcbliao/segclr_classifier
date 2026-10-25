#!/bin/bash
# Invoked only after newest-SegCLR feature assembly, so completed embedding
# arrays no longer consume the GPU QOS submission quota.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
source scripts/sbatch/latest_conf0_env.sh
mkdir -p "$CONF0_STATE"
record=$CONF0_STATE/jobs.json
if [[ -e "$record" ]]; then echo "Existing classifier submission: $record" >&2; exit 1; fi
jq -n '{status:"submitting",label_confidence_min:0,configurations:4,folds:5}' > "$record"
remember() {
  jq --arg key "$1" --arg value "$2" '. + {($key):$value}' "$record" > "$record.tmp"
  mv "$record.tmp" "$record"
}
preparation=$(sbatch --parsable scripts/sbatch/latest_conf0_cpu.sh prepare); preparation=${preparation%%;*}; remember preparation "$preparation"
training=$(sbatch --parsable --array=0-19%4 --dependency="afterok:$preparation" scripts/sbatch/latest_conf0_gpu.sh train); training=${training%%;*}; remember training_array "$training"
plan=$(sbatch --parsable --job-name=latest_conf0_inference_plan --dependency="afterok:$training" scripts/sbatch/latest_conf0_cpu.sh prepare-inference); plan=${plan%%;*}; remember inference_plan "$plan"
pilot=$(sbatch --parsable --job-name=latest_conf0_inference_parity --time=00:30:00 --cpus-per-task=4 --mem=16G --dependency="afterok:$plan" scripts/sbatch/latest_conf0_gpu.sh pilot); pilot=${pilot%%;*}; remember inference_parity "$pilot"
predictions=$(sbatch --parsable --job-name=latest_conf0_predictions --exclude=node3006 --array=0-31%4 --dependency="afterok:$pilot" scripts/sbatch/predict_fragment_cell_types.sh infer --output "$CONF0_OUTPUT"); predictions=${predictions%%;*}; remember prediction_array "$predictions"
audit=$(sbatch --parsable --job-name=latest_conf0_prediction_audit --dependency="afterok:$predictions" scripts/sbatch/latest_conf0_cpu.sh finish); audit=${audit%%;*}; remember prediction_audit "$audit"
preflight=$(sbatch --parsable --job-name=latest_conf0_import_preflight --export="ALL,SEGCLR_IMPORT_SOURCE=$CONF0_OUTPUT,SEGCLR_IMPORT_STATE=logs/presynaptic_import_latest100k_conf0" --dependency="afterok:$audit" scripts/sbatch/import_presynaptic_cell_types.sh preflight); preflight=${preflight%%;*}; remember import_preflight "$preflight"
import_job=$(sbatch --parsable --job-name=latest_conf0_import --export="ALL,SEGCLR_IMPORT_SOURCE=$CONF0_OUTPUT,SEGCLR_IMPORT_STATE=logs/presynaptic_import_latest100k_conf0" --dependency="afterok:$preflight" scripts/sbatch/import_presynaptic_cell_types.sh import); import_job=${import_job%%;*}; remember import_job "$import_job"
remember status submitted
jq --slurpfile classifiers "$record" '. + {classifier_jobs:$classifiers[0]}' analysis/db_deployment/presynaptic/latest_segclr/jobs.json > analysis/db_deployment/presynaptic/latest_segclr/jobs.tmp
mv analysis/db_deployment/presynaptic/latest_segclr/jobs.tmp analysis/db_deployment/presynaptic/latest_segclr/jobs.json
cat "$record"
