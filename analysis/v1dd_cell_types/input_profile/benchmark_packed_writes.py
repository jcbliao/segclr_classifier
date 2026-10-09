import sys,time,tempfile,json,os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from v1dd_packed_chunks import PackedCache,BufferedPackedCache
base='/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196'
items={'write_benchmark_'+str(i):bytes([i%256])*262144 for i in range(512)}
results={}
with tempfile.TemporaryDirectory(prefix='packed_write_benchmark_',dir=base) as folder:
 for name,cls in [('individual',PackedCache),('buffered',BufferedPackedCache)]:
  cache=cls(Path(folder)/name);started=time.monotonic()
  with patch('v1dd_packed_chunks.os.fsync',wraps=os.fsync) as sync:
   for key,payload in items.items():cache.put(key,payload)
   if hasattr(cache,'flush'):cache.flush()
   results[name]=dict(seconds=time.monotonic()-started,fsyncs=sync.call_count)
  fresh=PackedCache(Path(folder)/name)
  for key,payload in items.items():assert fresh.get(key)==payload
 results['speedup']=results['individual']['seconds']/results['buffered']['seconds']
 Path('analysis/v1dd_cell_types/input_profile/packed_write_benchmark.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(results),flush=True)
