#!/bin/bash
#SBATCH --job-name=prep_teasar111
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/prep_teasar111_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
segclr_db/.venv/bin/python - <<'PY'
import sys, json, hashlib, shutil
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, 'scripts')
from resample_teasar_dense import densify
# Branched tree plus isolated vertex: preserve cable, endpoints and topology.
v = np.array([[0,0,0], [1000,0,0], [1000,200,0], [1000,0,30], [3000,0,0]], dtype=np.float32)
e = np.array([[0,1],[1,2],[1,3]])
w, f, r = densify(v,e,np.arange(5))
np.testing.assert_array_equal(w[:len(v)],v)
l = np.linalg.norm(w[f[:,0]] - w[f[:,1]],axis=1)
assert l.max() <= 111.001
assert np.isclose(l.sum(),1230)
assert len(w)-len(f) == len(v)-len(e)
assert np.count_nonzero(f == 1) == 3
assert len(densify(v,np.empty((0,2),int),np.arange(5))[0]) == 5
for spacing in [0,-1,float('nan')]:
    try: densify(v,e,np.arange(5),spacing)
    except ValueError: pass
    else: raise AssertionError('invalid spacing accepted')
src = Path('/orcd/data/sdorkenw/001/collina/segclr_downstream_analysis/round_1/cells.csv')
other = Path('/orcd/data/sdorkenw/001/collina/downstream_output_full_nodes/round_1/cells.csv')
a,b = pd.read_csv(src),pd.read_csv(other)
pd.testing.assert_frame_equal(a,b)
assert a.pt_root_id.is_unique and len(a)==466
out = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_testing_teasar_111nm_20260910')
out.mkdir(parents=True,exist_ok=True)
shutil.copyfile(src,out/'cells.csv')
(out/'roots.txt').write_text(''.join(f'{x}\n' for x in a.pt_root_id))
(out/'provenance.json').write_text(json.dumps(dict(source=str(src),matching_source=str(other),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),counts=a.cell_type.value_counts().to_dict(),spacing_nm=111,pipeline='skeletonization.build_normal_offset_skeleton defaults',resampling='edge subdivision, ceil(length/111), original vertices preserved'),indent=2)+'\n')
print('Resampling checks passed. Cohort:',len(a),a.cell_type.value_counts().to_dict(),flush=True)
from segclr_db import store as st
print('Loaded database implementation:', st.__file__,flush=True)
PY
cd segclr_db
.venv/bin/python -m pytest tests/test_named_skeletons.py tests/test_skeletons.py -q
