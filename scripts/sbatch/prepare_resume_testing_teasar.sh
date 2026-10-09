#!/bin/bash
#SBATCH --job-name=prep_resume_teasar
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/prep_resume_teasar_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
segclr_db/.venv/bin/python - <<'PY'
from pathlib import Path
import json, shutil
import numpy as np
base = Path('/orcd/scratch/orcd/013/jcbliao/skeletons')
old = base/'segclr_testing_teasar_111nm_20260910'
existing = base/'segclr_testing_existing_teasar_111nm_20260910'
out = base/'segclr_testing_teasar_111nm_20260910_resumed'
out.mkdir(exist_ok=True)
for directory in ['skeletons','resampled_111nm']:
    (out/directory).mkdir(exist_ok=True)
for name in ['cells.csv','roots.txt']:
    shutil.copyfile(old/name,out/name)
rows=[]
for rid in (out/'roots.txt').read_text().split():
    source=base/'segclr/skeletons'/f'{rid}.npz'
    dense=existing/'resampled_111nm'/f'{rid}.npz'
    origin='existing_archive'
    if not source.exists():
        source=old/'skeletons'/f'{rid}.npz'
        dense=old/'resampled_111nm'/f'{rid}.npz'
        origin='interrupted_run'
    if source.exists():
        with np.load(source) as raw:
            assert int(raw['root_id'][0]) == int(rid)
            v,e,r=raw['vertices'],raw['edges'],raw['radius']
            assert len(v)==len(r) and np.isfinite(v).all()
            assert not e.size or (e.min()>=0 and e.max()<len(v))
        dest=out/'skeletons'/f'{rid}.npz'
        if not dest.exists(): dest.symlink_to(source)
        if dense.exists():
            with np.load(dense) as data:
                assert int(data['root_id'][0])==int(rid)
                np.testing.assert_array_equal(data['vertices'][:len(v)],v.astype(np.float32))
            dest=out/'resampled_111nm'/dense.name
            if not dest.exists(): dest.symlink_to(dense)
            report=dense.with_suffix('.json')
            if report.exists():
                target=dest.with_suffix('.json')
                if not target.exists(): target.symlink_to(report)
        rows.append(dict(root_id=rid,origin=origin,source=str(source),dense_reused=dense.exists()))
    else:
        rows.append(dict(root_id=rid,origin='needs_generation',dense_reused=False))
summary={k:sum(r['origin']==k for r in rows) for k in ['existing_archive','interrupted_run','needs_generation']}
summary['dense_reused']=sum(r['dense_reused'] for r in rows)
(out/'provenance.json').write_text(json.dumps(dict(summary=summary,cells=rows),indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
PY
