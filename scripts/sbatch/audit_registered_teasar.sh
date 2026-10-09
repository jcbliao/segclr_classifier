#!/bin/bash
#SBATCH --job-name=audit_registered_teasar
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/audit_registered_teasar_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/home/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
segclr_db/.venv/bin/python -u - <<'PY'
from pathlib import Path
import json
from segclr_db import store as st
base=Path('/orcd/scratch/orcd/013/jcbliao/skeletons')
out=base/'segclr_registered_teasar_111nm_20260911'
out.mkdir(exist_ok=True)
s=st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns')
roots=set(map(int,st.scan(s,'cells',columns=['root_id']).to_pydict()['root_id']))
presyn=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10')
priority={int(p.stem) for p in (presyn/'cells').glob('*.npz')}&roots
old=base/'segclr_testing_teasar_111nm_20260910_resumed'
rows=[]
for rid in sorted(roots):
    sources=[base/'segclr/skeletons'/f'{rid}.npz',old/'skeletons'/f'{rid}.npz']
    source=next((p for p in sources if p.exists()),None)
    rows.append(dict(root_id=rid,priority=rid in priority,source=str(source) if source else None,
                     previous_dense=str(old/'resampled_111nm'/f'{rid}.npz') if (old/'resampled_111nm'/f'{rid}.npz').exists() else None))
(out/'plan.json').write_text(json.dumps(rows,indent=2)+'\n')
for label,selected in [('priority',[r for r in rows if r['priority']]),('remaining',[r for r in rows if not r['priority']])]:
    (out/f'{label}_roots.txt').write_text(''.join(f"{r['root_id']}\n" for r in selected))
summary=dict(registered=len(roots),priority=len(priority),remaining=len(roots-priority),raw_available=sum(r['source'] is not None for r in rows),missing=[r['root_id'] for r in rows if not r['source']],previous_dense=sum(r['previous_dense'] is not None for r in rows))
(out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
