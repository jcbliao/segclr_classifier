"""Shared post-cache deduplication, target-root membership, and durable CPU batches."""
import sys,tempfile,multiprocessing as mp,threading,json
from collections import OrderedDict
from types import SimpleNamespace
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from v1dd_shared_post_crops import SharedPostCrops

def fake(base,count):
 c=SharedPostCrops.__new__(SharedPostCrops)
 c.raw=OrderedDict();c.raw_pending={};c.raw_bytes=0;c.raw_guard=threading.Lock();c.root_guard=threading.Lock();c.root_map={}
 c.chunk_lock=threading.Lock();c.chunk_pending={};c.shared_connections=set();c.connection_epoch=0;c.local=threading.local();c.timestamp=object()
 def roots(ids,timestamp):
  assert list(ids)==[7] and timestamp is c.timestamp
  return np.full(len(ids),101,np.uint64)
 c.resolution=np.ones(3)
 c.client=SimpleNamespace(chunkedgraph=SimpleNamespace(get_roots=roots,get_leaves=lambda root,bounds,stop_layer:np.array([7],np.uint64),base_resolution=np.ones(3)));c.select_shard(base,3)
 def volume(seg):
  def download(box,**kwargs):
   with count.get_lock():count.value+=1
   return np.full(tuple(np.asarray(box.maxpt)-box.minpt),7 if seg else 9,np.uint64 if seg else np.uint8)
  return SimpleNamespace(dtype=np.dtype(np.uint64 if seg else np.uint8),download=download,chunk_size=np.full(3,128),voxel_offset=np.zeros(3,dtype=int),bounds=SimpleNamespace(minpt=np.zeros(3,dtype=int),maxpt=np.full(3,2)))
 c._volumes=lambda:(volume(False),volume(True));return c

def child(base,count,rank):
 c=fake(base,count);em,seg=c._volumes();a=np.zeros(3,dtype=int);b=np.full(3,2,dtype=int)
 # Different download order exercises flush-before-wait deadlock prevention.
 for volume,is_seg in ([(em,False),(seg,True)] if rank%2==0 else [(seg,True),(em,False)]):
  x=c._raw_chunk(volume,a,b,is_seg);assert np.all(x==(7 if is_seg else 9))
 c.close_batch_connections()

with tempfile.TemporaryDirectory() as folder:
 count=mp.Value('i',0);children=[mp.Process(target=child,args=(folder,count,r)) for r in range(4)]
 for p in children:p.start()
 for p in children:p.join(20);assert not p.is_alive() and p.exitcode==0
 assert count.value==2,count.value
 c=fake(folder,count);c.prepare_targets([101]);image,mask=c.fetch_crop(np.ones(3,dtype=int),101)
 assert image.shape==(1,129,129,129) and int(mask.sum())==8
 assert count.value==2
 c.close_batch_connections()
 counter=Path(folder)/'chunks_v1196_v1/cells/3/cache_bytes.json';assert 0<json.loads(counter.read_text())<1024
 print('PASS cross-process image/root chunk sharing, retained ownership, cyclic-wait recovery, target-membership masks and bounded allocation')

# Exercise the real CPU batch publication and GPU output/cleanup lifecycle with
# a deterministic model stub; the model itself is unchanged production SegCLR.
import pandas as pd
import prepare_v1dd_post_batch_queue as prep
import infer_v1dd_prepared_post as gpu
from v1dd_batch_queue import BatchQueue
from v1dd_storage_budget import StorageBudget
from concurrent.futures import ThreadPoolExecutor
with tempfile.TemporaryDirectory() as folder:
 base=Path(folder);prep.BASE=base;gpu.BASE=base
 for name in ['sites','inputs_v1','embeddings','embedding_commits']:(base/name).mkdir()
 frame=pd.DataFrame(dict(synapse_id=[1001,1002],presynaptic_root_id=[900,900],postsynaptic_root_id=[101,0],x_nm=[1.,1.],y_nm=[1.,1.],z_nm=[1.,1.]))
 frame.to_parquet(base/'sites/part-00003.parquet',index=False)
 (base/'manifest.json').write_text('{}')
 StorageBudget(base).initialize(10*1024**3);q=BatchQueue(base);q.initialize([(3,2)]);q.planned(3,5*1024**3)
 claim=q.claim();assert claim[:3]==('batch',3,0)
 count=mp.Value('i',0);c=fake(base,count);c.resolution=np.ones(3)
 with ThreadPoolExecutor(max_workers=2) as pool:prep.prepare(q,c,3,0,frame,pool)
 claim[3].close();p=base/'inputs_v1/3/batch_000000'
 assert np.load(p/'statuses.npy').tolist()==['ok','unresolved_root']
 assert np.unpackbits(np.load(p/'mask.npy')[0],count=129**3).sum()==8
 prep.collect(q,3);assert (base/'inputs_v1/3/ready.json').exists()
 assert not (base/'chunks_v1196_v1/cells/3').exists()
 def engine(manifest,prepared):
  assert prepared
  def run(part,paths):
   ids=np.concatenate([np.load(p/'node_ids.npy') for p in paths]);return ids,np.ones((len(ids),64),np.float32)
  return SimpleNamespace(_run_prepared=run)
 gpu.make_inference=engine;gpu.infer(3)
 out=pd.read_parquet(base/'embeddings/part-00003.parquet')
 assert out.synapse_id.tolist()==[1001,1002] and out.status.tolist()==['ok','unresolved_root']
 assert out.embedding.iloc[1] is None
 assert not (base/'inputs_v1/3').exists() and (base/'embedding_commits/3.json').exists()
 assert q.snapshot(3)['status']=='DONE'
 assert json.loads((base/'storage_budget.json').read_text())['cells']['3']['bytes']==0
 print('PASS shared CPU publication, exact site/status alignment, whole-shard cache cleanup and GPU verified-output cleanup/release')

# Root mapping must be batched across sites, including different target roots.
with tempfile.TemporaryDirectory() as folder:
 count=mp.Value('i',0);c=fake(folder,count);calls=[]
 def grouped(ids,timestamp):
  calls.append(list(ids));return np.full(len(ids),101,np.uint64)
 c.client.chunkedgraph.get_roots=grouped
 image=np.zeros((8,1,129,129,129),np.uint8);mask=np.zeros((8,(129**3+7)//8),np.uint8);statuses=np.full(8,'ok',dtype='<U20')
 with ThreadPoolExecutor(max_workers=4) as pool:c.fill_batch(np.ones((8,3),dtype=int),np.array([101]*4+[102]*4),image,mask,statuses,pool)
 assert calls==[[7]],calls
 assert statuses.tolist()==['ok']*4+['empty_mask']*4
 assert all(np.unpackbits(m,count=129**3).sum()==8 for m in mask[:4])
 assert not mask[4:].any() and not image[4:].any()
 c.close_batch_connections()
 print('PASS one root lookup shared across eight crops, exact target masks and empty-image handling')

import requests

# Failed large requests split without changing ID ordering or masking semantics.
with tempfile.TemporaryDirectory() as folder:
 c=fake(folder,mp.Value('i',0));sizes=[]
 def fail_large(ids,timestamp):
  sizes.append(len(ids))
  if len(ids)>128:
   response=requests.Response();response.status_code=413
   raise requests.HTTPError(response=response)
  return np.asarray(ids,dtype=np.uint64)+1000
 c.client.chunkedgraph.get_roots=fail_large;ids=np.arange(1,514,dtype=np.uint64)
 np.testing.assert_array_equal(c._resolve_labels(ids),ids+1000)
 assert sizes[0]==513 and any(n<=128 for n in sizes)
 c.close_batch_connections()
 print('PASS large-request splitting preserves exact label/root order')

# Bounds must be clipped before encoding, and 64-bit IDs must remain exact.
with tempfile.TemporaryDirectory() as folder:
 c=fake(folder,mp.Value('i',0));large=2**57+3;seen=[]
 def bounded(root,bounds,stop_layer):
  seen.append(bounds.copy());assert np.all(bounds>=0)
  return np.array([large],dtype=np.int64)
 c.client.chunkedgraph.get_leaves=bounded
 candidates=c._bounded_candidates(np.zeros(3,dtype=int),101,np.array([large,large+1,0],np.uint64))
 np.testing.assert_array_equal(candidates,np.array([large],np.uint64));assert candidates.dtype==np.uint64
 np.testing.assert_array_equal(seen[0],np.array([[0,2],[0,2],[0,2]]))
 c.close_batch_connections()
 print('PASS bounded leaf queries clip padded crops and preserve IDs above 2**53')

# Public-snapshot root rejections are data exclusions, not token failures.
with tempfile.TemporaryDirectory() as folder:
 c=fake(folder,mp.Value('i',0));calls=[]
 def excluded(root,bounds,stop_layer):
  calls.append(root);response=requests.Response();response.status_code=401;response._content=b'root_id not valid at timestamp'
  raise requests.HTTPError(response=response)
 c.client.chunkedgraph.get_leaves=excluded
 assert c._bounded_candidates(np.zeros(3,dtype=int),101,np.array([7],np.uint64)).size==0
 assert c._bounded_candidates(np.ones(3,dtype=int),101,np.array([7],np.uint64)).size==0
 assert calls==[101]
 def unauthorized(root,bounds,stop_layer):
  response=requests.Response();response.status_code=401;response._content=b'Unauthorized token'
  raise requests.HTTPError(response=response)
 c.client.chunkedgraph.get_leaves=unauthorized
 try:c._bounded_candidates(np.zeros(3,dtype=int),102,np.array([7],np.uint64))
 except requests.HTTPError:pass
 else:raise AssertionError('Token failures must propagate')
 c.close_batch_connections()
 print('PASS public-invalid root exclusion is cached; authorization failures propagate')

# Overload responses retry the same payload; they must not multiply requests.
from unittest.mock import patch
import v1dd_api_gate
with tempfile.TemporaryDirectory() as folder:
 c=fake(folder,mp.Value('i',0));sizes=[]
 def overloaded_once(ids,timestamp):
  sizes.append(len(ids))
  if len(sizes)==1:
   response=requests.Response();response.status_code=502
   raise requests.HTTPError(response=response)
  return np.asarray(ids,dtype=np.uint64)+1000
 c.client.chunkedgraph.get_roots=overloaded_once;ids=np.arange(1,514,dtype=np.uint64)
 with patch.object(v1dd_api_gate.time,'sleep'):np.testing.assert_array_equal(c._resolve_labels(ids),ids+1000)
 assert sizes==[513,513],sizes
 c.close_batch_connections()
 print('PASS HTTP502 backs off on the same payload without request amplification')
