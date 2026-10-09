"""Instrument one production batch without changing its data or queue semantics."""
import sys,time,threading,collections,json,os,fcntl,sqlite3,argparse,traceback,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import prepare_v1dd_batch_queue as worker
from v1dd_shared_chunks import SharedCrops
from infer_v1dd_candidates import V1DDCrops
lock=threading.Lock();totals=collections.Counter();counts=collections.Counter();active={};started=time.monotonic();stop=threading.Event()
def timed(label,fn,*args,**kwargs):
 key=(threading.get_ident(),object());t=time.monotonic()
 with lock:active[key]=(label,t)
 try:return fn(*args,**kwargs)
 finally:
  with lock:totals[label]+=time.monotonic()-t;counts[label]+=1;active.pop(key,None)
def wrap(obj,name,label):
 original=getattr(obj,name)
 def call(*args,**kw):return timed(label,original,*args,**kw)
 setattr(obj,name,call)
original_flock=fcntl.flock
def flock(fd,operation):
 if operation & fcntl.LOCK_UN:return original_flock(fd,operation)
 try:name=os.readlink('/proc/self/fd/'+str(fd if isinstance(fd,int) else fd.fileno()))
 except OSError:name='unknown'
 label='lock.shard' if name.endswith('.access.lock') else ('lock.chunk' if '/locks/' in name else ('lock.queue' if 'batch_queue.lock' in name else 'lock.other'))
 return timed(label,original_flock,fd,operation)
fcntl.flock=flock
import v1dd_shared_chunks as cache_module
import v1dd_batch_queue as queue_module
import v1dd_storage_budget as budget_module
acquire=cache_module.acquire_lock
def acquire_profile(handle):
    name=os.readlink('/proc/self/fd/'+str(handle.fileno()))
    label='wait.shard' if name.endswith('.access.lock') else ('wait.chunk' if '/locks/' in name else ('wait.queue' if 'batch_queue.lock' in name else 'wait.other'))
    return timed(label,acquire,handle)
cache_module.acquire_lock=queue_module.acquire_lock=budget_module.acquire_lock=acquire_profile
connect=sqlite3.connect
class Connection:
 def __init__(self,c):self.c=c
 def execute(self,sql,*args,**kw):
  label='sqlite.select' if sql.startswith('SELECT') else 'sqlite.execute'
  return timed(label,self.c.execute,sql,*args,**kw)
 def commit(self):return timed('sqlite.commit',self.c.commit)
 def close(self):return timed('sqlite.close',self.c.close)
 def __getattr__(self,name):return getattr(self.c,name)
sqlite3.connect=lambda *a,**kw:Connection(timed('sqlite.connect',connect,*a,**kw))
volumes=V1DDCrops._volumes
def measured_volumes(self):
 result=volumes(self)
 for v,label in zip(result,['download.image','download.segmentation']):
  if not getattr(v,'_profile_wrapped',False):
   original=v.download
   def download(*a,_fn=original,_label=label,**kw):return timed(_label,_fn,*a,**kw)
   v.download=download;v._profile_wrapped=True
 return result
V1DDCrops._volumes=measured_volumes
wrap(V1DDCrops,'_mask','mask.compute');wrap(V1DDCrops,'_cached_block','crop.assemble_inclusive')
wrap(SharedCrops,'_shared_connection','cache.connection_inclusive');wrap(SharedCrops,'_disk_chunk','cache.chunk_inclusive')
wrap(worker.np,'save','prepared.save');wrap(worker,'load_plan','plan.load');wrap(worker,'prepare_batch','batch.total')
def snapshot():
 now=time.monotonic()
 with lock:
  times=dict(totals);pending=collections.Counter()
  for label,t in active.values():times[label]=times.get(label,0)+now-t;pending[label]+=1
  out=dict(elapsed=round(now-started,1),cpu_seconds=round(time.process_time(),2),max_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),thread_seconds={k:round(v,2) for k,v in sorted(times.items())},calls=dict(counts),inflight=dict(pending))
 print('PROFILE '+json.dumps(out),flush=True)
 Path(os.environ.get('V1DD_PROFILE_REPORT','/tmp/v1dd_profile.json')).write_text(json.dumps(out,indent=2))
def monitor():
 while not stop.wait(30):snapshot()
threading.Thread(target=monitor,daemon=True).start()
def direct_batch():
 queue=worker.BatchQueue(worker.BASE)
 manifest=json.loads((worker.BASE/'manifest.json').read_text());crops=SharedCrops(manifest)
 from concurrent.futures import ThreadPoolExecutor
 held=None
 try:
  while held is None:
   state=json.loads(queue.path.read_text())
   for rid,cell in sorted(state['cells'].items(),key=lambda item:item[1]['n_batches']-len(item[1]['done'])):
    if cell['status']!='OPEN' or not cell.get('admitted'):continue
    for index in range(cell['n_batches']):
     if index in cell['done']:continue
     held=queue.try_lock(f'{rid}_batch_{index:06d}')
     if held:break
    if held:break
   if not held:time.sleep(5)
  rid=int(rid);print(f'DIRECT PROFILE root={rid} batch={index}',flush=True)
  payload=worker.load_plan(queue,rid)
  with ThreadPoolExecutor(max_workers=16) as readers:worker.prepare_batch(queue,crops,payload,rid,index,readers)
 finally:
  crops.chunk_pool.shutdown(wait=True);crops.close_batch_connections()
  if held:held.close()
try:
 if '--direct' in sys.argv:direct_batch()
 else:worker.work(argparse.Namespace(max_seconds=600,limit_batches=1))
finally:stop.set();snapshot()
