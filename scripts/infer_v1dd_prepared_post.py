"""GPU-only inference of a complete CPU-prepared post-site shard."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_v1dd_post_batch_queue import BASE
from infer_postsynaptic_sites import atomic_json
from infer_v1dd_candidates import make_inference
from v1dd_storage_budget import StorageBudget
from v1dd_batch_queue import BatchQueue

def infer(part):
 dest=BASE/'embeddings'/f'part-{part:05d}.parquet';folder=BASE/'inputs_v1'/str(part)
 if not dest.exists():
  ready=json.loads((folder/'ready.json').read_text());m=json.loads((BASE/'manifest.json').read_text())
  paths=[folder/f'batch_{i:06d}' for i in range(ready['n_batches'])]
  digest=hashlib.sha256((BASE/'sites'/dest.name).read_bytes()).hexdigest()
  for path in paths:
   if json.loads((path/'metadata.json').read_text())['sites_sha256']!=digest:
    raise RuntimeError('Prepared post sites changed before GPU inference')
  engine=make_inference(m,prepared=True)
  started=time.monotonic();ids,vectors=engine._run_prepared(part,paths)
  f=pd.read_parquet(BASE/'sites'/dest.name);np.testing.assert_array_equal(ids,np.arange(len(f)))
  statuses=np.concatenate([np.load(p/'statuses.npy') for p in paths]);assert len(statuses)==len(f)
  f['status']=statuses;f['error']='';f['embedding']=pd.Series([v if s=='ok' else None for v,s in zip(vectors,statuses)],dtype=object)
  temp=dest.with_suffix('.tmp.parquet');f.to_parquet(temp,index=False);check=pd.read_parquet(temp)
  np.testing.assert_array_equal(check.synapse_id,f.synapse_id)
  good=np.flatnonzero(statuses=='ok')
  if len(good):np.testing.assert_array_equal(np.stack(check.iloc[good].embedding),vectors[good])
  temp.replace(dest);print(f'POST GPU DONE shard={part} sites={len(f)} seconds={time.monotonic()-started:.3f}',flush=True)
 else:
  f=pd.read_parquet(dest)
  if (~f.status.isin(['ok','empty_mask','unresolved_root'])).any():raise RuntimeError('Existing shard has failed crops')
 shutil.rmtree(folder,ignore_errors=True)
 if folder.exists():raise RuntimeError('Prepared post inputs cleanup incomplete')
 StorageBudget(BASE).release(part)
 BatchQueue(BASE).update(lambda s:s['cells'][str(part)].update(status='DONE'))
 atomic_json(BASE/'embedding_commits'/f'{part}.json',dict(part=part,n_sites=len(f)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--part',type=int,required=True);a=p.parse_args();infer(a.part)
