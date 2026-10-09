"""Offline cache integrity audit; remove only confirmed corrupt disposable SQLite files."""
import concurrent.futures,json,sqlite3,time
from pathlib import Path
BASE=Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')
REPORT=Path(__file__).with_name('corrupt_cache_cleanup_20261001.json')
state=json.loads((BASE/'batch_queue.json').read_text())
paths=[p for p in (BASE/'chunks_v1196_v1'/'cells').rglob('*.sqlite') if state['cells'].get(p.relative_to(BASE/'chunks_v1196_v1'/'cells').parts[0],{}).get('status')=='OPEN']
print(f'Auditing {len(paths)} databases with workers held',flush=True)
def check(path):
 start=time.time();connection=None
 try:
  connection=sqlite3.connect(str(path),timeout=10)
  result=[r[0] for r in connection.execute('PRAGMA quick_check(1)')]
  return dict(path=str(path),corrupt=result!=['ok'],result=result,seconds=round(time.time()-start,2))
 except sqlite3.DatabaseError as exc:
  corrupt=getattr(exc,'sqlite_errorcode',None) in (sqlite3.SQLITE_CORRUPT,sqlite3.SQLITE_NOTADB) or 'malformed' in str(exc)
  return dict(path=str(path),corrupt=corrupt,error=str(exc),seconds=round(time.time()-start,2))
 finally:
  if connection:connection.close()
results=[];removed_bytes=0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for result in pool.map(check,paths):
  if result['corrupt']:
   path=Path(result['path']);size=0
   for suffix in ('','-journal','-wal','-shm'):
    item=Path(str(path)+suffix)
    if item.exists():size+=item.stat().st_blocks*512;item.unlink()
   result['removed_bytes']=size;removed_bytes+=size
   print('REMOVED',result['path'],result.get('result',result.get('error')),flush=True)
  results.append(result)
  if len(results)%50==0:print('CHECKED',len(results),'/',len(paths),flush=True)
  REPORT.write_text(json.dumps(dict(checked=len(results),total=len(paths),removed_bytes=removed_bytes,results=results),indent=2))
print('FINISHED',len(results),'corrupt',sum(r['corrupt'] for r in results),'removed GiB',removed_bytes/1024**3,flush=True)
