"""Verify complete, finite, correctly keyed postsynaptic point embeddings."""
import json,argparse
from pathlib import Path
import numpy as np
import duckdb
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('/orcd/scratch/orcd/013/jcbliao/postsynaptic_point_embeddings'))
out=parser.parse_args().output
plan=json.loads((out/'plan.json').read_text())
c=duckdb.connect();c.execute('SET threads=4')
metadata=[]
for root in sorted({x['root_id'] for x in plan['chunks']}):
 expected=c.execute('select synapse_id,cell_x_nm,cell_y_nm,cell_z_nm from read_parquet(?) where cell_root_id=? order by synapse_id',[plan['source'],root]).fetchnumpy()
 for item in sorted((x for x in plan['chunks'] if x['root_id']==root),key=lambda x:x['start']):
  start,n=item['start'],item['count'];path=out/'parts'/str(root)/f'{start:08d}.npz'
  with np.load(path) as z:
   assert int(z['root_id'])==root
   np.testing.assert_array_equal(z['synapse_ids'],expected['synapse_id'][start:start+n])
   np.testing.assert_array_equal(z['positions_nm'],np.column_stack([expected['cell_'+a+'_nm'][start:start+n] for a in 'xyz']))
   assert z['embeddings'].shape==(n,64) and np.isfinite(z['embeddings']).all()
  metadata.append(dict(root_id=root,start=start,n_points=n,path=str(path)))
assert sum(x['n_points'] for x in metadata)==plan['n_points']
(out/'validated_manifest.json').write_text(json.dumps(dict(n_points=plan['n_points'],n_cells=plan['n_cells'],checkpoint=plan['checkpoint'],parts=metadata),indent=2)+'\n')
print('Validated',plan['n_points'],'embeddings across',plan['n_cells'],'cells',flush=True)
