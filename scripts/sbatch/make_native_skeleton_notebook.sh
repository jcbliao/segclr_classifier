#!/bin/bash
#SBATCH --job-name=make_native_notebook
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/jcbliao/rotation/segclr/gnn_classifier/logs/make_native_notebook_%j.out
set -euo pipefail
cd /home/jcbliao/rotation/segclr/gnn_classifier
segclr_db/.venv/bin/python - <<'PY'
import json
from pathlib import Path
src=Path('analysis/presynaptic/new_skeletons/skeleton_stats.ipynb')
dest=Path('analysis/presynaptic/new_skeletons_native/skeleton_stats.ipynb')
nb=json.loads(src.read_text())
for cell in nb['cells']:
    text=''.join(cell['source'])
    text=text.replace("/ 'new_skeletons'","/ 'new_skeletons_native'")
    text=text.replace('manifest=ROOT / "data" / "manifest.json"','manifest=skeleton_stats.MANIFEST')
    if cell['cell_type']=='code':
        cell['outputs']=[]; cell['execution_count']=None
    cell['source']=text.splitlines(keepends=True)
nb['cells'][0]['source']=['# Native full-resolution TEASAR skeleton statistics\n','\n',
    'Uses the **upsampled (~111 nm), un-subsampled** TEASAR skeletons for the new testing cohort, after the same 5 µm soma exclusion. The population is restricted to proofread-axon cells with soma coordinates. Each edge is counted once; no inference or embedding file is loaded.\n','\n',
    'Presynaptic sites are mapped directly to the nearest retained geometry node within 2 µm. These nodes will all carry embeddings after inference. Statistical fits, percentile bounds, PMFs, cell-type panels, and upstream branch/synapse counts reuse the original notebook implementation.\n','\n',
    'Data: `dense_teasar_multiscale_v2/native/cells`. See `data/DENSE_PRESYNAPTIC.md` for preparation.\n']
for cell in nb['cells']:
    if cell['cell_type']=='markdown':
        text=''.join(cell['source']).replace('`data/manifest.json`','the new cohort manifest')
        text=text.replace('nearest CAVE SegCLR node','nearest retained TEASAR node')
        text=text.replace('without requiring `new_valid_k_window`','without constructing embedding windows')
        cell['source']=text.splitlines(keepends=True)
dest.write_text(json.dumps(nb,indent=1)+'\n')
print(dest)
PY
