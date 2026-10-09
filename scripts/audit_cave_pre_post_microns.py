"""Check MICrONS post inference covers every valid K10 window before reuse."""
import json
import numpy as np
import pandas as pd
import duckdb
from prepare_cave_pre_post_v1dd_addition import SOURCE,MICRONS,OUTPUT
m=json.loads((SOURCE/'manifests/with_v1dd/fold0.json').read_text());c=duckdb.connect();c.execute('SET threads=4')
f=c.execute('select presynaptic_root_id,synapse_id from read_parquet(?)',[str(MICRONS/'embeddings/*.parquet')]).df()
sets={int(r):set(map(int,g.synapse_id)) for r,g in f.groupby('presynaptic_root_id')};missing=[];checked=0
for rid,info in m['cells'].items():
 if info['source_dataset']!='microns':continue
 with np.load(SOURCE/'cells'/f'{rid}.npz') as z:ids=set(map(int,z['cave_synapse_id'][z['cave_valid_k_window']]))
 absent=ids-sets.get(int(rid),set())
 if absent:missing.append(dict(root_id=rid,n_missing=len(absent),synapse_ids=sorted(absent)))
 checked+=1
OUTPUT.mkdir(parents=True,exist_ok=True);p=OUTPUT/'microns_coverage.json';p.write_text(json.dumps(dict(checked_cells=checked,missing=missing),indent=2))
print('MICrONS coverage:',checked,'cells;',len(missing),'cells with missing sites',flush=True)
if missing:raise RuntimeError('MICrONS post-site inference must cover the missing sites')
