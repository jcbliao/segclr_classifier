"""Submit all nine filtered runs and a dependent comparison summary."""
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[3]
script = 'scripts/sbatch/train_native_pre_post_axon_filtered.sh'
job = subprocess.check_output(['sbatch', '--parsable', '--array=0-8%2', script], cwd=ROOT, text=True).strip().split(';')[0]
output = ROOT / 'results/presynaptic/native_pre_post_axon_filtered'
output.mkdir(parents=True, exist_ok=True)
record = dict(training_array=int(job), tasks={str(i): f'{arch}/{condition}'
    for i, (condition, arch) in enumerate((c, a) for c in ('pre_only', 'pre_post', 'pre_mean_control')
    for a in ('mean', 'pointwise_mlp', 'graph_transformer'))},
    compartment_filter='/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/compartment_filtered/scale16/k17/conf0.7/fold0/geodesic_exclusion20um')
marker = output / 'submission.json'
marker.write_text(json.dumps(record, indent=2) + '\n')
summary = subprocess.check_output(['sbatch', '--parsable', f'--dependency=afterok:{job}',
    '--job-name=filtered_pre_post_summary', '--partition=mit_normal', '--account=mit_general', '--qos=normal',
    '--cpus-per-task=1', '--mem=2G', '--time=00:10:00', f'--chdir={ROOT}',
    f'--output={ROOT}/logs/%x_%j.out', f'--error={ROOT}/logs/%x_%j.err',
    '--wrap=segclr_db/.venv/bin/python analysis/presynaptic/native_pre_post_axon_filtered/summarize.py --include-controls'],
    cwd=ROOT, text=True).strip().split(';')[0]
record['comparison'] = int(summary)
marker.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
