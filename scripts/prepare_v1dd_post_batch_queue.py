"""Shared CPU queue: fixed 128 post-site crops, bounded shard lifecycle."""
import argparse,faulthandler,hashlib,json,os,shutil,signal,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from v1dd_batch_queue import BatchQueue
from v1dd_storage_budget import StorageBudget,prepared_bytes
from v1dd_shared_post_crops import SharedPostCrops,CACHE_CAP
from infer_postsynaptic_sites import atomic_json
BASE=Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1dd_v1196')

def initialize():
 for folder in ['inputs_v1','embedding_commits']:(BASE/folder).mkdir(exist_ok=True)
 q=BatchQueue(BASE);m=json.loads((BASE/'manifest.json').read_text())
 if q.path.exists():return
 StorageBudget(BASE).initialize(600*1024**3)
 counts=[]
 for i in range(m['n_shards']):
  f=pd.read_parquet(BASE/'sites'/f'part-{i:05d}.parquet');counts.append((i,len(f)))
 q.initialize(counts)
 for i,n in counts:
  q.planned(i,prepared_bytes(n)+CACHE_CAP+4*1024**2)
  if (BASE/'embeddings'/f'part-{i:05d}.parquet').exists():
   q.update(lambda s,i=i:s['cells'][str(i)].update(status='DONE'))
   atomic_json(BASE/'embedding_commits'/f'{i}.json',dict(part=i,recovered=True))
 print('POST QUEUE INITIALIZED',len(counts),'shards; budget 600 GiB',flush=True)

def collect(q,part,held=None):
 lock=held or q.try_lock(f'{part}_gc')
 if lock is None:return
 try:
  cell=q.snapshot(part)
  if cell['status']!='OPEN' or len(cell['done'])!=cell['n_batches']:return
  shutil.rmtree(BASE/'chunks_v1196_v1/cells'/str(part),ignore_errors=True)
  atomic_json(BASE/'inputs_v1'/str(part)/'ready.json',dict(part=part,n_nodes=cell['n_nodes'],n_batches=cell['n_batches']))
  q.budget.ready(part,prepared_bytes(cell['n_nodes']));q.gc_complete(part,cell['n_batches'],ready=True)
  print('POST SHARD READY',part,flush=True)
 finally:lock.close()

def prepare(q,crops,part,index,frame,readers):
 folder=BASE/'inputs_v1'/str(part);folder.mkdir(parents=True,exist_ok=True);target=folder/f'batch_{index:06d}'
 ids=np.arange(index*128,min((index+1)*128,len(frame)),dtype=np.int64)
 digest=hashlib.sha256((BASE/'sites'/f'part-{part:05d}.parquet').read_bytes()).hexdigest()
 if target.exists():
  assert json.loads((target/'metadata.json').read_text())['sites_sha256']==digest
  np.testing.assert_array_equal(np.load(target/'node_ids.npy'),ids);q.complete(part,index);return
 for temp in folder.glob(f'.batch_{index:06d}.*.tmp'):shutil.rmtree(temp)
 crops.select_shard(BASE,part);start=time.monotonic()
 image=np.zeros((len(ids),1,129,129,129),np.uint8);mask=np.zeros((len(ids),(129**3+7)//8),np.uint8)
 statuses=np.full(len(ids),'ok',dtype='<U20')
 roots=frame.postsynaptic_root_id.to_numpy(np.int64);centers=crops.nm_to_voxel(frame[['x_nm','y_nm','z_nm']].to_numpy())
 crops.fill_batch(centers[ids],roots[ids],image,mask,statuses,readers)
 if crops.raw_pending:raise RuntimeError('Raw chunk readers still active')
 crops.close_batch_connections()
 temp=folder/f'.batch_{index:06d}.{os.getpid()}.tmp';temp.mkdir()
 for name,array in [('image',image),('mask',mask),('node_ids',ids),('statuses',statuses)]:np.save(temp/f'{name}.npy',array,allow_pickle=False)
 atomic_json(temp/'metadata.json',dict(part=part,batch=index,sites_sha256=digest,n_nodes=len(ids),batch_size=128))
 temp.replace(target);q.complete(part,index)
 print(f'POST INPUT READY shard={part} batch={index} nodes={len(ids)} seconds={time.monotonic()-start:.3f}',flush=True)

def work(max_seconds):
 q=BatchQueue(BASE);m=json.loads((BASE/'manifest.json').read_text());crops=SharedPostCrops(m);frames={};deadline=time.monotonic()+max_seconds
 try:
  with ThreadPoolExecutor(max_workers=int(os.environ.get('V1DD_POST_CROP_THREADS','8'))) as readers:
   while time.monotonic()<deadline:
    claim=q.claim()
    if claim is None:time.sleep(5);continue
    kind,part,index,handle=claim
    if kind=='finished':return
    try:
     if kind=='finalize':collect(q,part,handle);handle=None;continue
     if kind=='setup':raise RuntimeError('Post queue plans must be initialized')
     if part not in frames:frames={part:pd.read_parquet(BASE/'sites'/f'part-{part:05d}.parquet')}
     prepare(q,crops,part,index,frames[part],readers)
    finally:
     if handle:handle.close()
    collect(q,part)
 finally:crops.close_batch_connections()
 raise SystemExit(75)
if __name__=='__main__':
 faulthandler.register(signal.SIGUSR2)
 p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--max-seconds',type=float,default=42000);a=p.parse_args()
 initialize() if a.initialize else work(a.max_seconds)
