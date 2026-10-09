"""Concurrency checks for shared batch claims and cache reclamation."""
import sys,tempfile,multiprocessing as mp,time,threading
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from v1dd_batch_queue import BatchQueue
from v1dd_storage_budget import StorageBudget
from v1dd_shared_chunks import SharedCrops,reclaim_chunks
from v1dd_chunk_lifetimes import ChunkLifetimes


def claimer(base,results,release):
    claim=BatchQueue(base).claim();results.put(claim[:3] if claim else None)
    release.wait(10)
    if claim and claim[3]:claim[3].close()

with tempfile.TemporaryDirectory() as folder:
    base=Path(folder);(base/'inputs_v1').mkdir();(base/'embeddings').mkdir()
    b=StorageBudget(base);b.initialize(10**9)
    q=BatchQueue(base);q.initialize([(11,1024)],window=8);q.planned(11,10000)
    results=mp.Queue();release=mp.Event();children=[mp.Process(target=claimer,args=(base,results,release)) for _ in range(8)]
    for child in children:child.start()
    claims=[results.get(timeout=10) for _ in children]
    assert {c[2] for c in claims}==set(range(8)) and all(c[:2]==('batch',11) for c in claims)
    assert q.claim() is None
    q.complete(11,7);assert q.prefix(q.snapshot(11))==0
    release.set()
    for child in children:child.join();assert child.exitcode==0
    claim=q.claim();assert claim[:2]==('batch',11) and claim[2] in range(7);claim[3].close()
    for i in reversed(range(7)):q.complete(11,i)
    assert q.prefix(q.snapshot(11))==8
    claim=q.claim();assert claim[:3]==('finalize',11,None);claim[3].close()
    q.gc_complete(11,8,ready=True);assert q.claim()[0]=='finished'
with tempfile.TemporaryDirectory() as folder:
    base=Path(folder);(base/'inputs_v1').mkdir();(base/'embeddings').mkdir()
    StorageBudget(base).initialize(10**9)
    q=BatchQueue(base);q.initialize([(11,128*20)]);q.planned(11,10000)
    claims=[q.claim() for _ in range(20)]
    assert {c[2] for c in claims}==set(range(20))
    assert q.claim() is None
    for c in claims:c[3].close()
print('PASS uncapped per-cell claims beyond eight workers')
print('PASS eight simultaneous workers claim distinct batches; out-of-order completion; abandoned claim recovery; finalization gating')


def fake_crops(base,shared_calls):
    c=SharedCrops.__new__(SharedCrops);c.disk_cache=Path(base)/'chunks_v1196_v1';(c.disk_cache/'locks').mkdir(parents=True,exist_ok=True)
    c.active_root=11;c.local=threading.local();c.chunk_lock=threading.Lock();c.chunk_pending={};c.batch_futures=None;c.chunk_cache=OrderedDict();c.chunk_bytes=0;c.chunk_budget=2*1024**3;c.shared_connections=set();c.connection_epoch=0;c.leaves={11:np.array([7],np.uint64)}
    def download(box,**kw):
        with shared_calls.get_lock():shared_calls.value+=1
        time.sleep(.1)
        return np.full(tuple(np.asarray(box.maxpt)-box.minpt),7,np.uint8)
    v=SimpleNamespace(dtype=np.dtype(np.uint8),download=download)
    return c,v


def downloader(base,count,results):
    c,v=fake_crops(base,count);data=c._disk_chunk(v,[0,0,0],[2,2,2],None)
    results.put(bool(np.all(data==7)));c.close_batch_connections()

with tempfile.TemporaryDirectory() as folder:
    count=mp.Value('i',0);results=mp.Queue();children=[mp.Process(target=downloader,args=(folder,count,results)) for _ in range(4)]
    for child in children:child.start()
    assert all(results.get(timeout=10) for _ in children)
    for child in children:child.join();assert child.exitcode==0
    assert count.value==1
    c,v=fake_crops(folder,count)
    v.chunk_size=np.array([2,2,2]);v.voxel_offset=np.zeros(3,dtype=int);v.bounds=SimpleNamespace(minpt=np.zeros(3,dtype=int),maxpt=np.full(3,2,dtype=int))
    c._volumes=lambda:(v,v)
    p=ChunkLifetimes(c,np.array([[1,1,1],[1,1,1]]),np.array([0,1]),batch_size=1,window=2)
    assert 'image:0_0_0_2_2_2' in p.items
    reclaim_chunks(folder,11,p,0,1)
    c._disk_chunk(v,[0,0,0],[2,2,2],None);assert count.value==1
    c.close_batch_connections();reclaim_chunks(folder,11,p,1,2)
    c._disk_chunk(v,[0,0,0],[2,2,2],None);assert count.value==2
    c.close_batch_connections()
print('PASS multi-process chunk reuse, retained future-use data, and reclamation only after all dependent batches complete')

# Already admitted batch claims must not wait for queue state mutations.
import fcntl
with tempfile.TemporaryDirectory() as folder:
    base=Path(folder);(base/'inputs_v1').mkdir();(base/'embeddings').mkdir()
    StorageBudget(base).initialize(10**9)
    q=BatchQueue(base);q.initialize([(11,128*3)]);q.planned(11,10000)
    initial=q.claim();initial[3].close()
    with q.lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        results=mp.Queue();release=mp.Event()
        child=mp.Process(target=claimer,args=(base,results,release));child.start()
        claim=results.get(timeout=5)
        assert claim[:2]==('batch',11)
        release.set();child.join(timeout=5);assert child.exitcode==0
    before=q.path.stat().st_mtime_ns
    q.snapshot(11);assert q.path.stat().st_mtime_ns==before
    before=q.budget.path.stat().st_mtime_ns
    assert not q.budget.reserve(12,2*10**9)
    assert q.budget.path.stat().st_mtime_ns==before
print('PASS admitted claims bypass global queue lock; read-only snapshots and denied reservations do not rewrite state')

# Different chunks in one shard must download simultaneously, while the earlier
# test still proves a duplicate chunk downloads only once across processes.
import zlib
seen={};pair=None
for i in range(1000):
    start=(i*2,0,0);end=(i*2+2,2,2)
    name='_'.join(map(str,start+end));key='image:'+name;shard=zlib.crc32(key.encode())%64
    stripe=zlib.crc32(('11:'+key).encode())%4096
    if shard in seen and seen[shard][2]!=stripe:
        pair=(seen[shard][:2],(start,end));break
    seen[shard]=(start,end,stripe)
assert pair is not None

def parallel_download(base,count,box,entered,release,results):
    c,v=fake_crops(base,count)
    def download(bounds,**kw):
        entered.set()
        assert release.wait(10)
        return np.full(tuple(np.asarray(bounds.maxpt)-bounds.minpt),7,np.uint8)
    v.download=download
    data=c._disk_chunk(v,box[0],box[1],None);results.put(bool(np.all(data==7)))
    c.close_batch_connections()

with tempfile.TemporaryDirectory() as folder:
    count=mp.Value('i',0);release=mp.Event();entered=[mp.Event(),mp.Event()];results=mp.Queue()
    children=[mp.Process(target=parallel_download,args=(folder,count,pair[i],entered[i],release,results)) for i in range(2)]
    children[0].start();assert entered[0].wait(5)
    children[1].start();concurrent=entered[1].wait(5);release.set()
    for child in children:child.join(timeout=10);assert child.exitcode==0
    assert concurrent,'A different chunk in the same shard waited for the first remote download'
    assert all(results.get(timeout=5) for _ in children)
    import sqlite3
    for db in (Path(folder)/'chunks_v1196_v1'/'cells'/'11'/'shared').glob('*.sqlite'):
        with sqlite3.connect(db) as conn:assert conn.execute('PRAGMA quick_check').fetchone()==('ok',)
print('PASS simultaneous remote downloads within one shard and resulting database integrity')

# One process now shares its SQLite connections among sixteen crop threads.
from concurrent.futures import ThreadPoolExecutor
with tempfile.TemporaryDirectory() as folder:
    count=mp.Value('i',0);c,v=fake_crops(folder,count)
    boxes=[((i*2,0,0),(i*2+2,2,2)) for i in range(40)]
    requests=boxes*3
    def read(box):return c._disk_chunk(v,box[0],box[1],None)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for data in pool.map(read,requests):assert np.all(data==7)
    assert count.value==len(boxes)
    assert len(c.shared_connections)<=64
    c.close_batch_connections()
    import sqlite3
    for db in (Path(folder)/'chunks_v1196_v1'/'cells'/'11'/'shared').glob('*.sqlite'):
        with sqlite3.connect(db) as conn:assert conn.execute('PRAGMA quick_check').fetchone()==('ok',)
print('PASS sixteen threads share cache connections safely, reuse duplicate chunks, and preserve SQLite integrity')
