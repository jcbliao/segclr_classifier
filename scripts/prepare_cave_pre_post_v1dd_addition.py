"""Prepare exact V1DD post sites and matched K10 pre/post caches."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from infer_postsynaptic_sites import CHECKPOINT,atomic_json
SOURCE=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_v1dd_addition')
OUTPUT=SOURCE.with_name('cave_pre_post_v1dd_addition')
SITES=Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1dd_v1196')
MICRONS=SITES.with_name('postsynaptic_site_embeddings_v1718')
INFERENCE=SITES.with_name('v1dd_candidates_v1196')

def prepare_sites():
 SITES.mkdir(parents=True,exist_ok=True)
 if (SITES/'manifest.json').exists():return
 cohort=pd.read_parquet(INFERENCE/'cohort.parquet');frames=[]
 for rid in cohort.pt_root_id:
  with np.load(SOURCE/'cells'/f'{rid}.npz') as z:
   ids=z['cave_synapse_id'][z['cave_valid_k_window']]
  f=pd.read_parquet(SOURCE/'sites'/f'{rid}.parquet');f=f[f.synapse_id.isin(ids)].copy()
  if len(f)!=len(set(map(int,ids))):raise ValueError(f'Missing paired sites for {rid}')
  frames.append(f)
 f=pd.concat(frames,ignore_index=True).rename(columns={'cell_root_id':'presynaptic_root_id','partner_root_id':'postsynaptic_root_id','partner_x_nm':'x_nm','partner_y_nm':'y_nm','partner_z_nm':'z_nm'})
 cols=['synapse_id','presynaptic_root_id','postsynaptic_root_id','partner_supervoxel_id','x_nm','y_nm','z_nm']
 f=f[cols].drop_duplicates()
 if f.synapse_id.duplicated().any():raise ValueError('Conflicting V1DD synapse IDs')
 if not np.isfinite(f[['x_nm','y_nm','z_nm']]).all().all():raise ValueError('Nonfinite post coordinates')
 f=f.sort_values(['z_nm','y_nm','x_nm','synapse_id']).reset_index(drop=True)
 shard=4096
 for folder in ['sites','embeddings']:(SITES/folder).mkdir(exist_ok=True)
 for i,start in enumerate(range(0,len(f),shard)):f.iloc[start:start+shard].to_parquet(SITES/'sites'/f'part-{i:05d}.parquet',index=False)
 source=json.loads((INFERENCE/'manifest.json').read_text())
 m=dict(datastack='v1dd_public',materialization_version=1196,synapse_table='synapses_v1dd',n_sites=len(f),n_shards=(len(f)+shard-1)//shard,shard_size=shard,n_unresolved=int(f.postsynaptic_root_id.eq(0).sum()),checkpoint_path=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),run_id=source['run_id'],experiment_id='resnet_860b_reshuffled',checkpoint_id=source['checkpoint_id'],embedding_dim=64,side='post',coordinate_units='nm',crop_size_voxels=129,precision='fp16',compiled=True,batch_size=128,dynamic=False,image_source=source['image_source'],segmentation_source=source['segmentation_source'],selection='all valid K10 presynaptic-window rows in active 265-cell cohort',masking='supervoxel roots resolved at materialization timestamp',sources=[str(SOURCE/'sites')])
 atomic_json(SITES/'manifest.json',m);print(json.dumps(m,indent=2),flush=True)

def prepare_cache():
 vm=json.loads((SITES/'manifest.json').read_text());vs=json.loads((SITES/'summary.json').read_text());mm=json.loads((MICRONS/'manifest.json').read_text());ms=json.loads((MICRONS/'summary.json').read_text())
 assert vs['n_sites']==vm['n_sites'] and ms['n_sites']==mm['n_sites']
 assert vm['checkpoint_sha256']==mm['checkpoint_sha256']
 manifests={a:json.loads((SOURCE/'manifests'/a/'fold0.json').read_text()) for a in ['with_v1dd']}
 roots=manifests['with_v1dd']['cells'];OUTPUT.mkdir(parents=True,exist_ok=True)
 for folder in ['cells','postsynaptic/cells']:(OUTPUT/folder).mkdir(parents=True,exist_ok=True)
 c=duckdb.connect();c.execute('SET threads=4');frames={}
 for domain,folder in [('microns',MICRONS),('v1dd',SITES)]:
  selected=pd.DataFrame({'presynaptic_root_id':[int(r) for r,i in roots.items() if i['source_dataset']==domain]})
  frame=c.execute('select e.* from read_parquet(?) e join selected using(presynaptic_root_id)',[str(folder/'embeddings/*.parquet')]).df()
  if frame.duplicated(['presynaptic_root_id','synapse_id']).any():raise ValueError('Duplicate paired embedding rows')
  if (~frame.status.isin(['ok','empty_mask','unresolved_root'])).any():raise ValueError('Failed crops must be repaired before cache publication')
  frames[domain]={int(r):rows.set_index('synapse_id') for r,rows in frame.groupby('presynaptic_root_id')}
 counts={'cells':0,'sites':0,'eligible_sites':0};inventory=[]
 for rid,info in roots.items():
  src=SOURCE/'cells'/f'{rid}.npz';dest=OUTPUT/'cells'/src.name
  if not dest.exists():dest.symlink_to(src)
  with np.load(src) as z:ids=z['cave_synapse_id'].copy();valid=z['cave_valid_k_window'].copy()
  rows=frames[info['source_dataset']].get(int(rid),pd.DataFrame())
  if len(rows):positions=rows.index.get_indexer(ids)
  else:positions=np.full(len(ids),-1)
  if np.any(valid & (positions<0)):raise ValueError(f'{rid}: valid windows lack post inference records')
  vectors=np.zeros((len(ids),64),np.float32);eligible=np.zeros(len(ids),bool);post_roots=np.zeros(len(ids),np.uint64);xyz=np.full((len(ids),3),np.nan)
  for j in np.flatnonzero(positions>=0):
   r=rows.iloc[positions[j]];post_roots[j]=int(r.postsynaptic_root_id);xyz[j]=[r.x_nm,r.y_nm,r.z_nm]
   if r.status=='ok' and r.postsynaptic_root_id!=0:vectors[j]=r.embedding;eligible[j]=True
  assert np.isfinite(vectors).all()
  p=OUTPUT/'postsynaptic/cells'/src.name;t=p.with_suffix('.tmp.npz')
  np.savez_compressed(t,synapse_ids=ids,eligible=eligible,post_x=vectors,postsynaptic_root_id=post_roots,postsynaptic_xyz_nm=xyz);t.replace(p)
  counts['cells']+=1;counts['sites']+=len(ids);counts['eligible_sites']+=int(eligible.sum());inventory.append(dict(root_id=rid,source_dataset=info['source_dataset'],n_valid_pre=int(valid.sum()),n_valid_pre_post=int((valid&eligible).sum())))
 for condition,m in manifests.items():
  m.update(source_database=str(OUTPUT),postsynaptic_cache=str(OUTPUT/'postsynaptic'))
  p=OUTPUT/'manifests'/condition/'fold0.json';p.parent.mkdir(parents=True,exist_ok=True);atomic_json(p,m)
 pd.DataFrame(inventory).to_csv(OUTPUT/'postsynaptic_inventory.csv',index=False)
 # Test the actual loader in both domains, with the same eligibility for every condition.
 sys.path.insert(0,str(REPO));from data.dataset_presynaptic import PresynapticWindowDataset
 for domain in ['microns','v1dd']:
  m=manifests['with_v1dd'];r=next(i['root_id'] for i in inventory if i['source_dataset']==domain and i['n_valid_pre_post']>0)
  # Marker is required by loader; publish only a temporary marker until probes pass.
  marker=OUTPUT/'postsynaptic/complete.json'
  atomic_json(marker,dict(format='postsynaptic-native-site-cache-v1',**counts,embedding_dim=64,checkpoint_sha256=vm['checkpoint_sha256']))
  try:
   probe={**m,'cells':{r:m['cells'][r]}};split=m['cells'][r]['split']
   ds=PresynapticWindowDataset(probe,split,'cave',database=OUTPUT,postsynaptic_cache=OUTPUT/'postsynaptic',use_postsynaptic=True)
   g=ds[0];assert g.x.shape==(10,64) and g.postsynaptic_embedding.shape==(1,64)
  except BaseException:marker.unlink(missing_ok=True);raise
 atomic_json(OUTPUT/'complete.json',dict(**counts,k_observed=10,folds=[0],source_database=str(SOURCE),v1dd_sites=str(SITES),microns_sites=str(MICRONS),conditions=['pre_only','pre_post','pre_mean_control'],eligibility='same resolved-root, nonempty-mask sites in all three conditions'))
 print(counts,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['sites','cache']);a=p.parse_args()
 prepare_sites() if a.stage=='sites' else prepare_cache()
