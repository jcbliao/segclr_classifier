from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import requests
sys.path.insert(0, '/orcd/home/002/jcbliao/rotation/skeletonization')
from skeletonization.precomputed import write_skeleton
from caveclient import CAVEclient

rid='864691134886828794'
local=Path(__file__).resolve().parent
base=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209')
window_path=base/f'training_local_center/scale32/k9/cells/{rid}.npz'
w=np.load(window_path)
g=np.load(str(w['geometry_path']))
native=np.load(base/f'native/cells/{rid}.npz')
assert np.array_equal(w['new_synapse_id'],native['new_synapse_id'])
valid=np.flatnonzero(w['new_valid_k_window'] & native['new_nearest_component_retained'])
i=int(valid[np.argmin(w['new_nearest_observed_distance_nm'][valid])])
sid=int(w['new_synapse_id'][i]); xyz=native['new_synapse_xyz_nm'][i].astype(float)
a,b=w['new_window_offsets'][i:i+2]; members=w['new_window_members'][a:b]
a,b=w['new_window_edge_offsets'][i:i+2]; pairs=w['new_window_edges'][a:b]; pairs=pairs[pairs[:,0]<pairs[:,1]]
pos=g['pos_nm']; edges=g['edges']; lengths=g['edge_length_nm']
graph=coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(pos),len(pos))).tocsr()
selected=set(); expected=0.
for a,b in pairs:
 start,end=int(members[a]),int(members[b])
 dist,pred=dijkstra(graph,indices=start,return_predecessors=True)
 assert np.isfinite(dist[end]); expected+=float(dist[end])
 node=end
 while node!=start:
  prev=int(pred[node]); assert prev>=0
  selected.add(tuple(sorted((node,prev)))); node=prev
full_edges=np.array(sorted(selected)); nodes=np.unique(full_edges)
window_edges=np.searchsorted(nodes,full_edges)
assert abs(expected-float(w['new_cable_length_nm'][i]))<1.
assert len(nodes)==len(full_edges)+1
raw=np.load(str(native['skeleton_source']))
out=Path('/orcd/scratch/orcd/013/jcbliao/neuroglancer/microns/presynaptic_windows')/rid/f'synapse_{sid}_scale32_k9'
url='https://g-ffa18e.d1c26e.5898.data.globus.org'+str(out).split('/neuroglancer')[1]
write_skeleton(out/'skeleton',int(rid),raw['vertices'],raw['edges'])
write_skeleton(out/'window',int(rid),pos[nodes],window_edges)
dims={x:[1e-9,'m'] for x in 'xyz'}
def layer(name,source,color,width):
 return dict(type='segmentation',name=name,source='precomputed://'+source,segments=[rid],segmentColors={rid:color},skeletonRendering=dict(shader='void main() { emitDefault(); }',mode2d='lines_and_points',mode3d='lines',lineWidth3d=width))
mesh=layer('Cell mesh',f'https://g-ffa18e.d1c26e.5898.data.globus.org/microns/neurons/{rid}/mesh','#a6bddb',1)
mesh['objectAlpha']=0.15
state=dict(dimensions=dims,position=xyz.tolist(),crossSectionScale=25,projectionScale=45000,showSlices=False,layout='3d',layers=[mesh,layer('Your native TEASAR skeleton',url+'/skeleton','#5599ff',1),layer('Presynaptic window — scale32 k9',url+'/window','#ff5500',5),dict(type='annotation',name=f'Presynaptic point {sid}',source=dict(url='local://annotations',transform=dict(outputDimensions=dims)),annotationColor='#ffff00',annotations=[dict(type='point',id=str(sid),point=xyz.tolist(),description=f'Presynaptic site {sid}; root {rid}')])])
record=dict(root_id=rid,synapse_id=str(sid),point_nm=xyz.tolist(),distance_to_skeleton_nm=float(w['new_nearest_observed_distance_nm'][i]),window_source=str(window_path),skeleton_source=str(native['skeleton_source']),scale=32,k=9,window_vertices=len(nodes),window_edges=len(full_edges),cable_length_nm=expected,public_url=url)
for dest in [local,out]:
 dest.mkdir(parents=True,exist_ok=True)
 (dest/'state.json').write_text(json.dumps(state,indent=2))
 (dest/'metadata.json').write_text(json.dumps(record,indent=2))
for suffix in ['skeleton/info','skeleton/'+rid,'window/info','window/'+rid]:
 response=requests.get(url+'/'+suffix,timeout=30); response.raise_for_status()
 assert response.content==(out/suffix).read_bytes(),suffix
 print('Verified public asset:',suffix,flush=True)
client=CAVEclient('minnie65_public')
state_id=client.state.upload_state_json(state)
link=client.state.build_neuroglancer_url(state_id,ngl_url='https://spelunker.cave-explorer.org',target_site='spelunker')
(local/'link.txt').write_text(link+'\n'); (out/'link.txt').write_text(link+'\n')
print(json.dumps(record,indent=2)); print(link,flush=True)
