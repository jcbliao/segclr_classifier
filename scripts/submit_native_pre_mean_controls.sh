#!/bin/bash
# Three additional controls on the requested preemptable partition.
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
existing=$(squeue -h -u "$USER" -n native_pre_post -o %A | sort -u | paste -sd : -)
controls=$(sbatch --parsable --partition=mit_preemptable --account=mit_general --qos=normal --array=6-8%2 scripts/sbatch/train_native_pre_post.sh)
summary=$(sbatch --parsable --dependency="afterok:${existing:+${existing}:}${controls}" scripts/sbatch/summarize_native_pre_post.sh --include-controls)
export PRE_MEAN_CONTROL_JOB="$controls" PRE_MEAN_SUMMARY_JOB="$summary"
segclr_db/.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
p=Path('results/presynaptic/native_skeletons_pre_post/submission.json')
m=json.loads(p.read_text())
m.update(pre_mean_control_array=int(os.environ['PRE_MEAN_CONTROL_JOB'].split(';')[0]),
         comparison=int(os.environ['PRE_MEAN_SUMMARY_JOB'].split(';')[0]))
m['tasks'].update({'6': 'mean/pre_mean_control', '7': 'pointwise_mlp/pre_mean_control', '8': 'graph_transformer/pre_mean_control'})
p.write_text(json.dumps(m,indent=2)+'\n')
print('Queued controls:',m['pre_mean_control_array'],'comparison:',m['comparison'])
PY
