"""Shared per-cell chunk caches sharded to permit writes from multiple nodes."""
import fcntl
from contextlib import contextmanager
from v1dd_file_lock import acquire_lock
import logging
import sqlite3
import threading
import zlib
from pathlib import Path
import numpy as np
from infer_v1dd_candidates import V1DDCrops

SHARDS=64

class SharedCrops(V1DDCrops):
    def __init__(self,manifest):
        super().__init__(manifest);self.shared_connections=set();self.connection_epoch=0

    def _shared_connection(self,path):
        # All callers hold this database's access lock. Share one connection
        # between chunk threads instead of reopening it in sixteen thread LRUs.
        name=str(path)
        with self.chunk_lock:
            mapping=getattr(self,'shared_connection_map',None)
            if mapping is None:self.shared_connection_map=mapping={}
            if name in mapping:return mapping[name]
        new=not path.exists() or not path.stat().st_size
        connection=sqlite3.connect(path,timeout=120,check_same_thread=False)
        connection.execute('PRAGMA synchronous=NORMAL')
        if new:
            # Cache files are deleted whole; pointer-map/vacuum maintenance
            # provides no benefit and adds writes to every inserted chunk.
            connection.execute('CREATE TABLE chunks (key TEXT PRIMARY KEY, data BLOB NOT NULL)')
            connection.commit()
        with self.chunk_lock:
            self.shared_connections.add(connection);mapping[name]=connection
        return connection

    def close_batch_connections(self):
        with self.chunk_lock:
            if self.chunk_pending:raise RuntimeError('Chunk readers must finish before publishing a batch')
            connections=list(self.shared_connections);self.shared_connections.clear();self.shared_connection_map={};self.database_locks={};self.connection_epoch+=1
            self.batch_futures=None
        for connection in connections:connection.close()

    def select_root(self,rid,plan):
        if getattr(self,'active_root',None)!=rid:
            self.close_batch_connections()
            with self.chunk_lock:self.chunk_cache.clear();self.chunk_bytes=0
            self.prepare_root(rid)
        self.lifetimes=plan

    def _cache_operation(self,path,operation):
        # Only short SQLite operations run under this shard lock. In particular,
        # downloads and waiting for another chunk's downloader happen outside it.
        path.parent.mkdir(parents=True,exist_ok=True)
        with self.chunk_lock:
            guards=getattr(self,'database_locks',None)
            if guards is None:self.database_locks=guards={}
            guard=guards.setdefault(str(path),threading.Lock())
        # Also serialize threads explicitly; NFS implements flock differently
        # from local filesystems, and this connection is shared within a process.
        with guard:
            with path.with_suffix('.access.lock').open('a') as handle:
                acquire_lock(handle)
                return operation(self._shared_connection(path))

    def _read_cache(self,path,key):
        def read(connection):
            row=connection.execute('SELECT data FROM chunks WHERE key=?',(key,)).fetchone()
            return None if row is None else row[0]
        return self._cache_operation(path,read)

    def _write_cache(self,path,key,payload):
        def write(connection):
            connection.execute('INSERT OR IGNORE INTO chunks VALUES (?,?)',(key,payload))
            connection.commit()
        self._cache_operation(path,write)

    @contextmanager
    def _chunk_ownership(self,key):
        stripe=zlib.crc32(f'{int(self.active_root)}:{key}'.encode())%16384
        with (self.disk_cache/'locks'/f'poll_{stripe}.lock').open('a') as lock:
            acquire_lock(lock)
            yield lock

    def _disk_chunk(self,volume,start,end,root_id):
        from cloudvolume.lib import Bbox
        from cloudvolume.exceptions import EmptyVolumeException
        rid=int(self.active_root);shape=tuple(map(int,np.asarray(end)-start))
        name='_'.join(map(str,tuple(start)+tuple(end)));key=('mask:' if root_id is not None else 'image:')+name
        shard=zlib.crc32(key.encode())%SHARDS
        path=self.disk_cache/'cells'/str(rid)/'shared'/f'{shard:02d}.sqlite'
        def decode(payload):
            if root_id is not None:return np.unpackbits(np.frombuffer(payload,np.uint8),count=int(np.prod(shape))).reshape(shape).view(bool)
            return np.frombuffer(payload,dtype=volume.dtype).reshape(shape)
        cached=self._read_cache(path,key)
        if cached is not None:return decode(cached)
        with self._chunk_ownership(key):
            cached=self._read_cache(path,key)
            if cached is not None:return decode(cached)
            # Read legacy caches without modifying them or holding the new
            # shard lock. Whole-cell deletion is the only production cleanup.
            old=self.disk_cache/'cells'/str(rid)/'chunks.sqlite'
            old_payload=self._read_cache(old,key) if old.exists() else None
            legacy_mask=self.disk_cache/'mask_mip2'/str(rid)/str(zlib.crc32(name.encode())%256)/f'{name}.npy'
            if old_payload is not None:data=decode(old_payload)
            elif root_id is not None and legacy_mask.exists():data=np.unpackbits(np.load(legacy_mask),count=int(np.prod(shape))).reshape(shape).view(bool)
            else:
                try:data=volume.download(Bbox(start,end),agglomerate=False) if root_id is not None else volume.download(Bbox(start,end))
                except EmptyVolumeException:
                    if root_id is None:raise
                    logging.warning('Missing sparse segmentation root=%s bbox=%s',rid,name);data=np.zeros(shape,dtype=volume.dtype)
                data=np.asarray(data)
                if data.ndim==4:data=data[...,0]
                if root_id is not None:data=self._mask(data,rid)
            payload=np.packbits(data).tobytes() if root_id is not None else data.tobytes()
            self._write_cache(path,key,payload)
            return data


def reclaim_chunks(base,rid,plan,old_prefix,new_prefix):
    """The completed prefix proves every dependent batch is durable."""
    base=Path(base);cache=base/'chunks_v1196_v1';keys=[]
    for i in range(old_prefix,new_prefix):keys.extend(plan.expired.get(i,[]))
    if not keys:return 0
    grouped={}
    for key in keys:grouped.setdefault(zlib.crc32(key.encode())%SHARDS,[]).append(key)
    reclaimed=0
    databases=[(cache/'cells'/str(rid)/'shared'/f'{shard:02d}.sqlite',values) for shard,values in grouped.items()]
    databases.append((cache/'cells'/str(rid)/'chunks.sqlite',keys))
    for path,values in databases:
        if not path.exists():continue
        before=path.stat().st_size
        connection=sqlite3.connect(path,timeout=120)
        try:
            mode=connection.execute('PRAGMA auto_vacuum').fetchone()[0]
            if mode:connection.execute('PRAGMA auto_vacuum=FULL')
            connection.executemany('DELETE FROM chunks WHERE key=?',[(k,) for k in values]);connection.commit()
            if mode:connection.execute('PRAGMA auto_vacuum=INCREMENTAL')
        finally:connection.close()
        reclaimed+=max(0,before-path.stat().st_size)
    for key in keys:
        if key.startswith('mask:'):
            name=key[5:];p=cache/'mask_mip2'/str(rid)/str(zlib.crc32(name.encode())%256)/f'{name}.npy'
            try:reclaimed+=p.stat().st_blocks*512;p.unlink()
            except FileNotFoundError:pass
    return reclaimed
