#!/bin/bash
#SBATCH --job-name=refresh_teasar_infer
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/refresh_teasar_infer_%j.out
set -euo pipefail
REPO=/home/jcbliao/rotation/segclr/gnn_classifier
export PYTHONPATH=/home/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export OUT=/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910_resumed
"$REPO/segclr_db/.venv/bin/python" -u - <<'PY'
import os, json, subprocess
from pathlib import Path
from segclr_db import store as st
from segclr_db.skeletons import SkeletonCache
out=Path(os.environ['OUT'])
cache=SkeletonCache(st.open_store('/orcd/compute/sdorkenw/001/segclr-db','microns'))
roots=[int(x) for x in (out/'roots.txt').read_text().split()]
names={'skeletons':'teasar_testing_20260910','resampled_111nm':'teasar_testing_111nm_20260910'}
rows=cache.list_named_skeletons()
known=set(zip(rows.root_id,rows.skeleton_name))
todo=[r for r in roots if any((out/d/f'{r}.npz').exists() and (r,n) not in known for d,n in names.items())]
path=out/'refresh_ingestion_roots.txt'
path.write_text(''.join(f'{r}\n' for r in todo))
print(f'Cells with newly completed skeleton outputs: {len(todo)}',flush=True)
if todo:
    env=dict(os.environ,ROOTS_FILE=str(path))
    subprocess.run(['/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python','-u',
                    '/home/jcbliao/rotation/segclr/gnn_classifier/scripts/ingest_testing_teasar.py',
                    '--allow-partial'],env=env,check=True)
rows=cache.list_named_skeletons()
available=set(int(r) for r in rows.loc[rows.skeleton_name==names['resampled_111nm'],'root_id'])
assigned=set(int(x) for x in (out/'inference_ready_roots.txt').read_text().split())
new=sorted((available & set(roots))-assigned)
(out/'inference_refresh_roots.txt').write_text(''.join(f'{r}\n' for r in new))
summary=dict(newly_ingested_cells=len(todo),new_inference_cells=len(new),previously_assigned=len(assigned),dense_available=len(available & set(roots)))
(out/'inference_refresh_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
PY
