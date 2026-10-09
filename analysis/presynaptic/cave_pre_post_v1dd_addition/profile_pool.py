"""Two-worker shared preparation experiment, isolated from production."""
import argparse,faulthandler,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'scripts'))
import prepare_v1dd_post_batch_queue as prep
from v1dd_shared_post_crops import SharedPostCrops
from v1dd_batch_queue import BatchQueue
from v1dd_storage_budget import StorageBudget
SOURCE=Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1dd_v1196')
p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--validate',action='store_true');p.add_argument('--directory',default='debug_pool_batched_20261002');args=p.parse_args()
BASE=SOURCE/args.directory
prep.BASE=BASE
if args.initialize:
 BASE.mkdir(exist_ok=True);(BASE/'sites').mkdir(exist_ok=True)
 frame=pd.read_parquet(SOURCE/'sites/part-00000.parquet').iloc[:512].copy()
 frame.to_parquet(BASE/'sites/part-00000.parquet',index=False)
 StorageBudget(BASE).initialize(30*1024**3);q=BatchQueue(BASE);q.initialize([(0,len(frame))]);q.planned(0,25*1024**3)
 (BASE/'started.json').write_text(json.dumps(dict(time=time.time())))
 print('POOL INITIALIZED 512 sites,4 batches');raise SystemExit
q=BatchQueue(BASE)
if args.validate:
 assert len(q.snapshot(0)['done'])==4
 result=[]
 for index in range(4):
  folder=BASE/'inputs_v1/0'/f'batch_{index:06d}';statuses=np.load(folder/'statuses.npy');images=np.load(folder/'image.npy',mmap_mode='r');masks=np.load(folder/'mask.npy',mmap_mode='r')
  np.testing.assert_array_equal(np.load(folder/'node_ids.npy'),np.arange(index*128,(index+1)*128))
  assert images.shape==(128,1,129,129,129)
  assert masks.shape==(128,(129**3+7)//8)
  assert set(statuses)<=set(['ok','empty_mask','unresolved_root'])
  for row,status in enumerate(statuses):
   if status=='ok':assert masks[row].any()
  if index<4:
   previous=SOURCE/'debug_pool_batched_20261002/inputs_v1/0'/f'batch_{index:06d}'
   for name in ['statuses','mask','image']:np.testing.assert_array_equal(np.load(folder/f'{name}.npy',mmap_mode='r'),np.load(previous/f'{name}.npy',mmap_mode='r'))
  result.append(dict(batch=index,statuses={str(x):int(sum(statuses==x)) for x in np.unique(statuses)}))
 out=dict(batches=result,previous_512_sites_match_exactly=True)
 (BASE/'validation.json').write_text(json.dumps(out,indent=2));print('POOL VALIDATED',json.dumps(out));raise SystemExit
faulthandler.dump_traceback_later(120,repeat=True)
t=time.monotonic();crops=SharedPostCrops(json.loads((SOURCE/'manifest.json').read_text()));print('POOL INIT DONE seconds',time.monotonic()-t,flush=True)
frame=pd.read_parquet(BASE/'sites/part-00000.parquet')
try:
 with ThreadPoolExecutor(max_workers=8) as readers:
  while True:
   claim=q.claim()
   if claim is None:time.sleep(1);continue
   kind,part,index,handle=claim
   if kind!='batch':
    if handle:handle.close()
    break
   try:prep.prepare(q,crops,part,index,frame,readers)
   finally:handle.close()
 print('POOL WORKER DONE seconds',time.monotonic()-t,flush=True)
finally:crops.close_batch_connections();faulthandler.cancel_dump_traceback_later()
