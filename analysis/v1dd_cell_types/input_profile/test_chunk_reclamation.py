"""Focused checks for last-use eviction, restart reuse, and physical file shrink."""
import sys,tempfile,threading,json,zlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from collections import OrderedDict
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from infer_v1dd_candidates import V1DDCrops
from v1dd_chunk_lifetimes import ChunkLifetimes
from v1dd_storage_budget import StorageBudget

volumes=[SimpleNamespace(chunk_size=np.array(s),voxel_offset=np.array([3,5,7]),bounds=SimpleNamespace(minpt=np.array([3,5,7]),maxpt=np.array([2048,2048,2048])),dtype=np.uint8) for s in [(64,64,64),(256,256,32)]]
c=V1DDCrops.__new__(V1DDCrops);c._volumes=lambda:volumes
centers=np.array([[100,100,100],[500,600,700],[101,100,100],[1500,1400,1300],[1800,1700,1600],[1900,1800,1700]])
order=np.arange(len(centers));plan=ChunkLifetimes(c,centers,order,batch_size=2)
uses={}
for row,xyz in enumerate(centers):
    for volume,packed in zip(volumes,[False,True]):
        for lo,hi in c.chunk_boxes(volume,xyz):
            key=('mask:' if packed else 'image:')+'_'.join(map(str,tuple(lo)+tuple(hi)))
            uses.setdefault(key,set()).add(row//2)
assert set(uses)==set(plan.items)
for key,batches in uses.items():
    assert plan.first[key]==min(batches) and plan.last[key]==max(batches)
assert plan.peak_chunk_bytes<plan.full_cache_bytes
for batch in range(plan.n_batches):
    assert set(plan.batch_keys[batch])=={k for k,u in uses.items() if batch in u}

with tempfile.TemporaryDirectory() as folder:
    c.disk_cache=Path(folder);(c.disk_cache/'locks').mkdir();c.active_root=11;c.local=threading.local();c.chunk_lock=threading.Lock();c.chunk_pending={};c.batch_futures=None;c.chunk_cache=OrderedDict();c.chunk_bytes=0;c.cache_connections=[]
    c.configure_reclamation(plan);conn=c._cache_connection()
    payloads={key:bytes([i%256])*262144 for i,key in enumerate(plan.items)}
    conn.executemany('INSERT INTO chunks VALUES (?,?)',list(payloads.items()));conn.commit()
    db=c.disk_cache/'cells'/'11'/'chunks.sqlite';before=db.stat().st_size
    shared=next(k for k,v in uses.items() if 0 in v and 1 in v)
    c.expire_batch(0)
    retained=dict(conn.execute('SELECT key,data FROM chunks'))
    assert shared in retained
    for key,payload in retained.items():assert payload==payloads[key] and plan.last[key]>0
    assert db.stat().st_size<before
    c.expire_batch(0) # restart/skip already-completed batch is idempotent
    assert dict(conn.execute('SELECT key,data FROM chunks'))==retained
    c.expire_batch(1)
    assert shared not in dict(conn.execute('SELECT key,data FROM chunks'))
    c.expire_batch(2);assert conn.execute('SELECT count(*) FROM chunks').fetchone()[0]==0
    assert db.stat().st_size<32768
    c.chunk_pending={'live':1}
    try:c.expire_batch(0)
    except RuntimeError:pass
    else:raise AssertionError('Active readers must block eviction')
    c.chunk_pending={};c.release_root(11)

with tempfile.TemporaryDirectory() as folder:
    b=StorageBudget(folder);b.initialize(100)
    assert b.reserve(1,80);assert not b.reserve(2,30)
    assert b.reserve(1,40);assert b.reserve(2,30)
    state=json.loads(b.path.read_text());assert sum(x['bytes'] for x in state['cells'].values())==70
print('PASS exact last-use geometry, future reuse, durable-batch restart, physical disk reclamation, reader guard, and reduced reservations')
from concurrent.futures import ThreadPoolExecutor
with tempfile.TemporaryDirectory() as folder:
    c=V1DDCrops.__new__(V1DDCrops)
    c.disk_cache=Path(folder);(c.disk_cache/'locks').mkdir();c.active_root=11;c.local=threading.local();c.chunk_lock=threading.Lock();c.chunk_pending={};c.batch_futures=None;c.chunk_cache=OrderedDict();c.chunk_bytes=0;c.chunk_budget=2*1024**3;c.cache_connections=[];c.chunk_pool=ThreadPoolExecutor(max_workers=4);c.leaves={11:np.array([7],np.uint64)}
    def volume(shape,dtype):
        def download(box,**kw):return np.full(tuple(np.asarray(box.maxpt)-box.minpt),7,dtype)
        return SimpleNamespace(chunk_size=np.array(shape),voxel_offset=np.zeros(3,dtype=int),bounds=SimpleNamespace(minpt=np.zeros(3,dtype=int),maxpt=np.full(3,2048,dtype=int)),dtype=np.dtype(dtype),download=download)
    em,seg=volume((64,64,64),np.uint8),volume((256,256,32),np.uint64)
    c._volumes=lambda:(em,seg)
    xyz=np.array([[100,100,100],[500,500,500],[101,100,100],[700,700,700]])
    plan=ChunkLifetimes(c,xyz,np.arange(len(xyz)),batch_size=1);c.configure_reclamation(plan)
    for batch,center in enumerate(xyz):
        c.prefetch_batch(batch)
        image=c._cached_block(em,center);mask=c._cached_block(seg,center,11)
        assert image.shape==(129,129,129) and np.all(image==7)
        assert mask.shape==(129,129,129) and mask.dtype==bool and mask.all()
        c.expire_batch(batch)
    c.chunk_pool.shutdown();c.release_root(11)
print('PASS batch prefetch and image/mask reconstruction remain exact across shared-chunk last-use deletion')
