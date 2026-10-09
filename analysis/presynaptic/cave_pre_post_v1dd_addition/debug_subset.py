"""Isolated real 128-site preparation validation; never touches production queue."""
import faulthandler,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'scripts'))
import prepare_v1dd_post_batch_queue as prep
from v1dd_shared_post_crops import SharedPostCrops
from v1dd_batch_queue import BatchQueue
from v1dd_storage_budget import StorageBudget
faulthandler.dump_traceback_later(120,repeat=True)
source=Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1dd_v1196')
base=source/'debug_subset_20261002';base.mkdir(exist_ok=True);(base/'sites').mkdir(exist_ok=True)
frame=pd.read_parquet(source/'sites/part-00000.parquet').iloc[:256].copy()
frame.to_parquet(base/'sites/part-00000.parquet',index=False)
prep.BASE=base;StorageBudget(base).initialize(30*1024**3);q=BatchQueue(base)
if not q.path.exists():q.initialize([(0,len(frame))]);q.planned(0,25*1024**3)
t=time.monotonic();print('SUBSET INIT START',flush=True)
c=SharedPostCrops(json.loads((source/'manifest.json').read_text()));print('SUBSET INIT DONE seconds',time.monotonic()-t,flush=True)
try:
 with ThreadPoolExecutor(max_workers=8) as pool:
  for _ in range(2):
   claim=q.claim()
   if claim is None:raise RuntimeError('Subset claim missing')
   kind,part,index,handle=claim
   if kind!='batch':
    if handle:handle.close()
    break
   try:prep.prepare(q,c,part,index,frame,pool)
   finally:handle.close()
 c.close_batch_connections()
 results=[]
 for index in range(2):
  folder=base/'inputs_v1/0'/f'batch_{index:06d}';statuses=np.load(folder/'statuses.npy')
  image=np.load(folder/'image.npy',mmap_mode='r');masks=np.load(folder/'mask.npy',mmap_mode='r');ids=np.load(folder/'node_ids.npy')
  assert image.shape==(128,1,129,129,129) and masks.shape==(128,(129**3+7)//8)
  np.testing.assert_array_equal(ids,np.arange(index*128,(index+1)*128))
  assert set(statuses)<=set(['ok','unresolved_root','empty_mask'])
  for row,status in enumerate(statuses):
   if status=='ok':assert np.any(masks[row])
  results.append(dict(batch=index,n_sites=len(ids),statuses={str(x):int(sum(statuses==x)) for x in np.unique(statuses)}))
 result=dict(elapsed_seconds=time.monotonic()-t,batches=results)
 (base/'validation.json').write_text(json.dumps(result,indent=2));print('SUBSET VALIDATED',json.dumps(result),flush=True)
finally:c.close_batch_connections();faulthandler.cancel_dump_traceback_later()
