#!/bin/bash
#SBATCH --job-name=audit_existing_teasar
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/audit_existing_teasar_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
segclr_db/.venv/bin/python - <<'PY'
from pathlib import Path
import json, shutil
base = Path('/orcd/scratch/orcd/013/jcbliao/skeletons')
old = base/'segclr_testing_teasar_111nm_20260910'
out = base/'segclr_testing_existing_teasar_111nm_20260910'
source = base/'segclr/skeletons'
out.mkdir(exist_ok=True)
for name in ['cells.csv','roots.txt']:
    shutil.copyfile(old/name,out/name)
roots = (out/'roots.txt').read_text().split()
present = [r for r in roots if (source/f'{r}.npz').is_file()]
missing = [r for r in roots if r not in set(present)]
(out/'available_roots.txt').write_text(''.join(r+'\n' for r in present))
(out/'missing_roots.txt').write_text(''.join(r+'\n' for r in missing))
report = dict(source=str(source),cohort_size=len(roots),available=len(present),missing=missing,
              resampling='subdivide existing edges to at most 111 nm; preserve original vertices')
(out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
