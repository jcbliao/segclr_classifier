#!/bin/bash
# Submit feature work now and the classifier arrays after it completes, respecting
# the GPU QOS's pending-job cap. Every submitted job ID is saved immediately.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
state=analysis/db_deployment/presynaptic/latest_segclr
mkdir -p "$state" logs
record=$state/jobs.json
if [[ -e "$record" ]]; then echo "Existing submission recorded in $record" >&2; exit 1; fi
jq -n '{status:"submitting",label_confidence_min:0,segclr_run:"resnet50_1280b_slab1280_50um_mac_partial_aug_2_style_bf16_100k__20261022_140504",segclr_checkpoint:"checkpoint_e0_s100000"}' > "$record"
remember() {
  jq --arg key "$1" --arg value "$2" '. + {($key):$value}' "$record" > "$record.tmp"
  mv "$record.tmp" "$record"
}
if [[ $# -gt 0 ]]; then points=$1; else
  points=$(sbatch --parsable scripts/sbatch/latest_segclr_features_cpu.sh plan); points=${points%%;*}
fi
remember point_plan "$points"
smoke=$(sbatch --parsable --job-name=latest_segclr_parity --time=00:30:00 --dependency="afterok:$points" scripts/sbatch/latest_segclr_features_gpu.sh smoke); smoke=${smoke%%;*}; remember segclr_preprocessing_parity "$smoke"
embeddings=$(sbatch --parsable --array=0-31%4 --dependency="afterok:$smoke" scripts/sbatch/latest_segclr_features_gpu.sh embed); embeddings=${embeddings%%;*}; remember embedding_array "$embeddings"
assembly=$(sbatch --parsable --job-name=latest_segclr_assemble --dependency="afterok:$embeddings" scripts/sbatch/latest_segclr_features_cpu.sh assemble); assembly=${assembly%%;*}; remember feature_assembly "$assembly"
dispatch=$(sbatch --parsable --dependency="afterok:$assembly" scripts/sbatch/latest_conf0_dispatch.sh); dispatch=${dispatch%%;*}; remember classifier_dispatch "$dispatch"
remember status features_submitted
cat "$record"
