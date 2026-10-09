"""Checks durable append-only storage, reader/writer concurrency and recovery."""
import sys,tempfile,multiprocessing as mp,zlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from v1dd_packed_chunks import PackedCache

def writer(folder,rank):
    cache=PackedCache(folder)
    with ThreadPoolExecutor(max_workers=8) as pool:
        def save(i):
            key='chunk_'+str(i);payload=(key.encode()+b'x'*1000)
            cache.put(key,payload);assert cache.get(key)==payload
        list(pool.map(save,range(100)))

with tempfile.TemporaryDirectory() as folder:
    children=[mp.Process(target=writer,args=(folder,r)) for r in range(4)]
    for child in children:child.start()
    for child in children:child.join(timeout=30);assert child.exitcode==0
    cache=PackedCache(folder)
    for i in range(100):
        key='chunk_'+str(i);assert cache.get(key)==key.encode()+b'x'*1000
    records=[]
    for path in Path(folder).glob('*.index'):records.extend(json.loads(l) for l in path.read_text().splitlines())
    assert len(records)==100
    # Simulate interrupted data and index writes. The durable index remains the
    # source of truth, and the next writer removes only an unfinished last line.
    key='chunk_1';shard=zlib.crc32(key.encode())%64
    with (Path(folder)/f'{shard:02d}.data').open('ab') as f:f.write(b'orphaned')
    with (Path(folder)/f'{shard:02d}.index').open('ab') as f:f.write(b'["partial",123')
    cache=PackedCache(folder);assert cache.get(key)==key.encode()+b'x'*1000
    other=next('new_'+str(i) for i in range(10000) if zlib.crc32(('new_'+str(i)).encode())%64==shard)
    cache.put(other,b'new valid bytes');assert PackedCache(folder).get(other)==b'new valid bytes'
    # Damage one payload and require a miss, then append a valid replacement.
    record=next(r for r in records if r[0]==key)
    with (Path(folder)/f'{shard:02d}.data').open('r+b') as f:f.seek(record[1]);f.write(b'BAD')
    cache=PackedCache(folder);assert cache.get(key) is None
    cache.put(key,b'repaired bytes');assert PackedCache(folder).get(key)==b'repaired bytes'
print('PASS multiprocess/thread duplicate writers; immutable bulk reads; interrupted publication; corrupt extent detection and rebuild')

from v1dd_packed_chunks import BufferedPackedCache
from unittest.mock import patch
import os
with tempfile.TemporaryDirectory() as folder:
    cache=BufferedPackedCache(folder,max_bytes=1024*1024)
    payloads={'buffer_'+str(i):bytes([i%256])*1000 for i in range(256)}
    with patch('v1dd_packed_chunks.os.fsync',wraps=os.fsync) as sync:
        for key,payload in payloads.items():cache.put(key,payload)
        assert sync.call_count==0
        assert cache.get('buffer_0')==payloads['buffer_0']
        assert PackedCache(folder).get('buffer_0') is None
        cache.flush();assert sync.call_count<=128
    assert cache.pending_bytes==0
    for key,payload in payloads.items():assert PackedCache(folder).get(key)==payload
    cache.put('retry',b'valid')
    with patch('v1dd_packed_chunks.os.fsync',side_effect=OSError('injected flush failure')):
        try:cache.flush()
        except OSError:pass
        else:raise AssertionError('flush must propagate failure')
    assert cache.get('retry')==b'valid';cache.flush()
    assert PackedCache(folder).get('retry')==b'valid'
    small=BufferedPackedCache(folder,max_bytes=10);small.put('auto',b'12345678901')
    assert small.pending_bytes==0;assert PackedCache(folder).get('auto')==b'12345678901'
print('PASS buffered visibility, grouped fsync count, threshold flush and failed flush retry')

def buffered_writer(folder,rank):
    cache=BufferedPackedCache(folder,max_bytes=8000)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i:cache.put('shared_'+str(i),b'x'*1000),range(100)))
    cache.flush()
with tempfile.TemporaryDirectory() as folder:
    children=[mp.Process(target=buffered_writer,args=(folder,r)) for r in range(4)]
    for child in children:child.start()
    for child in children:child.join(timeout=30);assert child.exitcode==0
    records=[json.loads(l) for p in Path(folder).glob('*.index') for l in p.read_text().splitlines()]
    assert len(records)==100
    for i in range(100):assert PackedCache(folder).get('shared_'+str(i))==b'x'*1000
print('PASS buffered multiprocess/thread duplicate publication')
