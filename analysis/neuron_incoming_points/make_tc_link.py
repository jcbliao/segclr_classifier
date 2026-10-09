"""Create a viewer state labeling every incoming point of the top TC cell."""
import sys,json
from pathlib import Path
import numpy as np
import duckdb
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from make_neuroglancer_prediction_link import viewer_state
from data.synapses import build_client
out=Path(__file__).resolve().parent
c=duckdb.connect()
p=str(out/'points.parquet')
root,n=c.execute("select cell_root_id,count(*) n from read_parquet(?) where list_contains(cell_types,'thalamocortical') group by 1 order by n desc,cell_root_id limit 1",[p]).fetchone()
d=c.execute('select * from read_parquet(?) where cell_root_id=? order by synapse_id',[p,root]).df()
xyz=d[['cell_x_nm','cell_y_nm','cell_z_nm']].to_numpy(dtype=float)
state=viewer_state('',{'run':'thalamocortical_incoming'},['rainbow'],'',[],{'rainbow':''},0)
state['layers']=state['layers'][:2]
state['layers'][1].update(segments=[str(root)],selectedAlpha=.4,notSelectedAlpha=0,objectAlpha=.5)
annotations=[]
for r in d.itertuples(index=False):
 annotations.append(dict(type='point',id=str(r.synapse_id),point=[float(getattr(r,'cell_'+a+'_nm'))/scale for a,scale in zip('xyz',[8,8,40])],description=f'Incoming synapse {r.synapse_id}; presynaptic root {r.partner_root_id}; mat 1718'))
state['layers'].append(dict(type='annotation',name=f'Incoming postsynaptic points ({n})',annotationColor='#ff3300',annotations=annotations,visible=True))
state['selectedLayer']={'layer':state['layers'][-1]['name'],'visible':True}
state['position']=((xyz.min(axis=0)+xyz.max(axis=0))/2/[8,8,40]).tolist()
state['projectionScale']=max(30000,float(np.linalg.norm(np.ptp(xyz,axis=0)))*1.25)
state['showSlices']=True
assert len(annotations)==n and len(set(a['id'] for a in annotations))==n
assert np.allclose(np.array([a['point'] for a in annotations])*[8,8,40],xyz)
(out/'top_thalamocortical.state.json').write_text(json.dumps(state,indent=2)+'\n')
secret=json.loads((Path.home()/'.cloudvolume/secrets/global.daf-apis.com-cave-secret.json').read_text())
client=build_client(secret['token'])
state_id=client.state.upload_state_json(state)
url=client.state.build_neuroglancer_url(state_id,ngl_url='https://spelunker.cave-explorer.org/')
(out/'top_thalamocortical_link.txt').write_text(url+'\n')
print('Root:',root,'points:',n,'URL:',url,flush=True)
