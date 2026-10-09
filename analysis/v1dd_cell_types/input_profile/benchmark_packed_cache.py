"""Compare bulk packed reads against the live SQLite cache on the same keys."""
import sys,json,time,threading,random,shutil,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from v1dd_shared_chunks import SharedCrops
from v1dd_packed_chunks import PackedCache
B=Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')
s=json.loads((B/'batch_queue.json').read_text());c=SharedCrops.__new__(SharedCrops)
c.disk_cache=B/'chunks_v1196_v1';c.chunk_lock=threading.Lock();c.chunk_pending={};c.shared_connections=set();c.connection_epoch=0
samples=[]
for rid,cell in sorted(s['cells'].items(),key=lambda item:len(item[1]['done'])):
 if cell['status']!='OPEN' or not cell.get('admitted') or cell['n_batches']-len(cell['done'])<30:continue
 for path in (c.disk_cache/'cells'/rid/'shared').glob('*.sqlite'):
  keys=c._cache_operation(path,lambda conn:conn.execute('SELECT key FROM chunks LIMIT 10').fetchall())
  for row in keys:samples.append((path,row[0]))
  if len(samples)>=120:break
 if len(samples)>=120:break
assert len(samples)>20
folder=Path(tempfile.mkdtemp(prefix='cache_benchmark_',dir=B))
try:
 pack=PackedCache(folder)
 for path,key in samples:
  payload=c._read_cache(path,key);pack.put(key,payload);assert pack.get(key)==payload
 rng=random.Random(128);queries=samples*4;rng.shuffle(queries)
 # Warm up each backend once; run packed first and SQLite second to avoid
 # treating SQLite's first-touch population as its measured read cost.
 for path,key in samples:assert c._read_cache(path,key)==pack.get(key)
 t=time.monotonic()
 for path,key in queries:assert pack.get(key) is not None
 packed=time.monotonic()-t
 t=time.monotonic()
 for path,key in queries:assert c._read_cache(path,key) is not None
 sqlite=time.monotonic()-t
 out=dict(chunks=len(samples),reads=len(queries),packed_seconds=packed,sqlite_seconds=sqlite,read_speedup=sqlite/packed)
 print(json.dumps(out),flush=True)
 Path(__file__).with_name('packed_cache_read_benchmark.json').write_text(json.dumps(out,indent=2))
finally:c.close_batch_connections();shutil.rmtree(folder)
