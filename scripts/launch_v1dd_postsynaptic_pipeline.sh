#!/bin/bash
# Default V1DD post-site launch: shared CPU preparation, then prepared GPU jobs.
set -euo pipefail
cd /orcd/home/002/jcbliao/rotation/segclr/gnn_classifier
if squeue -u jcbliao -h -o '%j' | rg -q '^v1dd_post_(batch|dispatch)$'; then
 echo 'Shared postsynaptic preparation/dispatch is already active; refusing duplicate launch.' >&2
 exit 1
fi
segclr_db/.venv/bin/python scripts/prepare_v1dd_post_batch_queue.py --initialize
normal=$(sbatch --parsable --array="0-$((${V1DD_POST_NORMAL_WORKERS:-64}-1))" scripts/sbatch/prepare_v1dd_post_batch_queue.sh)
preempt=$(sbatch --parsable --partition=mit_preemptable --account=mit_general --qos=normal --array="0-$((${V1DD_POST_PREEMPTABLE_WORKERS:-428}-1))" scripts/sbatch/prepare_v1dd_post_batch_queue.sh)
segclr_db/.venv/bin/python - "$normal" "$preempt" <<'PYCODE'
import json,sys
from pathlib import Path
Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1dd_v1196/preparation_jobs.json').write_text(json.dumps({'job_ids':sys.argv[1:]}))
PYCODE
dispatch=$(sbatch --parsable scripts/sbatch/dispatch_v1dd_post_jobs.sh)
printf 'normal_preparation=%s\npreemptable_preparation=%s\ngpu_dispatch=%s\n' "$normal" "$preempt" "$dispatch"
