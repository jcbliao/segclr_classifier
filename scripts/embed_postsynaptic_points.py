"""Resumable pretrained SegCLR inference at every incoming synapse coordinate."""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

DEFAULT_OUTPUT=Path('/orcd/scratch/orcd/013/jcbliao/postsynaptic_point_embeddings')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
 p.add_argument('--source',type=Path,default=Path('analysis/neuron_incoming_points/points.parquet'))
 p.add_argument('--plan',action='store_true')
 p.add_argument('--task-id',type=int,default=0)
 p.add_argument('--num-tasks',type=int,default=64)
 p.add_argument('--limit',type=int)
 args=p.parse_args(); out=args.output
 if args.plan:
  import duckdb
  from scripts.check_embedding_augmentation_stability import CHECKPOINT
  source=args.source.resolve()
  c=duckdb.connect();c.execute('SET threads=4')
  counts=c.execute('select cell_root_id,count(*) n from read_parquet(?) group by 1 order by 1',[str(source)]).fetchall()
  chunks=[dict(root_id=int(root),start=start,count=min(4096,n-start)) for root,n in counts for start in range(0,n,4096)]
  loads=[0]*args.num_tasks
  for item in sorted(chunks,key=lambda x:-x['count']):
   rank=min(range(args.num_tasks),key=loads.__getitem__);item['task_id']=rank;loads[rank]+=item['count']
  out.mkdir(parents=True,exist_ok=True)
  assert not (out/'plan.json').exists(),'Plan already exists; use the existing plan to resume'
  plan=dict(source=str(source),source_size=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,checkpoint=str(CHECKPOINT),materialization_version=1718,point_columns=['cell_x_nm','cell_y_nm','cell_z_nm'],embedding_dim=64,num_tasks=args.num_tasks,n_points=sum(n for _,n in counts),n_cells=len(counts),chunks=chunks)
  (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  print('Planned',plan['n_points'],'points in',len(chunks),'chunks; rank loads',min(loads),max(loads),flush=True);return
 plan=json.loads((out/'plan.json').read_text()); source=Path(plan['source'])
 assert source.stat().st_size==plan['source_size'] and source.stat().st_mtime_ns==plan['source_mtime_ns'],'Input changed since planning'
 import duckdb
 from scripts.generate_cave_embedding_augmentations import FastAugInference,_atomic_savez,CHECKPOINT
 assert str(CHECKPOINT)==plan['checkpoint']
 engine=FastAugInference('clean',0,0,batch_size=32,num_threads=8)
 c=duckdb.connect();c.execute('SET threads=2')
 items=[x for x in plan['chunks'] if x['task_id']==args.task_id]
 if args.limit: items=items[:args.limit]
 started=time.time()
 for item in items:
  root=item['root_id'];start=item['start'];n=item['count']
  dest=out/'parts'/str(root)/f'{start:08d}.npz'
  if dest.exists():
   with np.load(dest) as z:
    assert z['embeddings'].shape==(n,64) and np.isfinite(z['embeddings']).all()
   continue
  d=c.execute('select synapse_id,cell_x_nm,cell_y_nm,cell_z_nm from read_parquet(?) where cell_root_id=? order by synapse_id limit ? offset ?',[str(source),root,n,start]).df()
  assert len(d)==n
  xyz=d.iloc[:,1:].to_numpy(dtype=np.float32); ids=np.arange(n,dtype=np.int64)
  got,emb=engine.embed(root,xyz,ids)
  np.testing.assert_array_equal(got,ids)
  assert emb.shape==(n,64) and np.isfinite(emb).all()
  _atomic_savez(dest,root_id=np.asarray(root,np.uint64),synapse_ids=d.synapse_id.to_numpy(dtype=np.int64),positions_nm=xyz,embeddings=emb.astype(np.float32))
  print(json.dumps(dict(root_id=root,start=start,n_points=n,seconds_elapsed=round(time.time()-started,1),output=str(dest))),flush=True)
 print('Task complete',args.task_id,flush=True)
if __name__=='__main__':main()
