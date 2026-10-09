"""Append-only chunk packs: immutable bulk reads, serialized durable publication."""
import json
import fcntl
import random
import time
from contextlib import contextmanager
import logging
import os
import threading
import zlib
from pathlib import Path
from v1dd_file_lock import acquire_lock
from v1dd_shared_chunks import SharedCrops,SHARDS

class PackedCache:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.guard=threading.Lock();self.states={}

    def _state(self,key):
        shard=zlib.crc32(key.encode())%SHARDS
        with self.guard:
            if shard not in self.states:
                self.states[shard]=dict(records={},offset=0,guard=threading.RLock(),writer=threading.Lock())
            return shard,self.states[shard]

    def _refresh(self,shard,state):
        path=self.folder/f'{shard:02d}.index'
        try:
            with path.open('rb') as handle:
                handle.seek(state['offset']);tail=handle.read()
        except FileNotFoundError:return
        end=tail.rfind(b'\n')+1
        for line in tail[:end].splitlines():
            try:
                key,offset,size,checksum=json.loads(line)
                if not isinstance(key,str) or any(type(v) is not int for v in [offset,size,checksum]):raise ValueError('invalid index types')
                if offset<0 or not 0<=size<=64*1024**2 or not 0<=checksum<=0xffffffff:raise ValueError('invalid extent')
                state['records'][key]=(offset,size,checksum)
            except (ValueError,TypeError):logging.warning('Ignoring malformed packed-cache index record in %s',path)
        # Ignore an interrupted trailing publication until it is completed or
        # truncated by a later writer holding this shard's publication lock.
        state['offset']+=end

    def get(self,key):
        shard,state=self._state(key)
        with state['guard']:
            if key not in state['records']:self._refresh(shard,state)
            record=state['records'].get(key)
        if record is None:return None
        offset,size,checksum=record
        try:
            # Open ensures NFS close-to-open coherence for newly appended data;
            # pread fetches the complete immutable chunk in one bulk operation.
            with (self.folder/f'{shard:02d}.data').open('rb',buffering=0) as handle:
                payload=os.pread(handle.fileno(),size,offset)
        except FileNotFoundError:payload=b''
        if len(payload)!=size or zlib.crc32(payload)!=checksum:
            logging.warning('Invalid packed-cache extent shard=%s key=%s; rebuilding chunk',shard,key)
            with state['guard']:state['records'].pop(key,None)
            return None
        return payload

    @staticmethod
    def _write_all(handle,payload):
        view=memoryview(payload)
        while view:
            n=handle.write(view)
            if not n:raise OSError('Short packed-cache write')
            view=view[n:]

    def put(self,key,payload):
        self.put_many({key:payload})

    def put_many(self,items):
        """Publish each shard with one data flush followed by one index flush."""
        groups={}
        for key,payload in items.items():
            shard,state=self._state(key)
            groups.setdefault(shard,[]).append((key,payload))
        for shard,entries in groups.items():
            _,state=self._state(entries[0][0])
            with state['writer']:
                with (self.folder/f'{shard:02d}.writer.lock').open('a') as lock:
                    acquire_lock(lock)
                    with state['guard']:self._refresh(shard,state)
                    entries=[(key,payload) for key,payload in entries if self.get(key) is None]
                    if not entries:continue
                    records=[]
                    with (self.folder/f'{shard:02d}.data').open('ab',buffering=0) as handle:
                        for key,payload in entries:
                            offset=handle.tell();self._write_all(handle,payload)
                            records.append((key,(offset,len(payload),zlib.crc32(payload))))
                        os.fsync(handle.fileno())
                    lines=b''.join((json.dumps([key,*record],separators=(',',':'))+'\n').encode()
                                   for key,record in records)
                    with state['guard']:
                        index=self.folder/f'{shard:02d}.index'
                        with index.open('r+b' if index.exists() else 'w+b',buffering=0) as handle:
                            handle.truncate(state['offset']);handle.seek(state['offset'])
                            self._write_all(handle,lines);os.fsync(handle.fileno())
                        state['records'].update(records);state['offset']+=len(lines)


class BufferedPackedCache(PackedCache):
    """Bounded process-local writes, made durable before batch publication.

    Download ownership may be transferred into the buffer and is released only
    after durable publication. Writers without ownership still deduplicate.
    """
    def __init__(self,folder,max_bytes=64*1024**2):
        super().__init__(folder)
        if max_bytes<=0:raise ValueError('Packed write buffer must be positive')
        self.max_bytes=max_bytes;self.pending={};self.pending_bytes=0;self.owners={}
        self.buffer_guard=threading.RLock()

    def get(self,key):
        with self.buffer_guard:
            payload=self.pending.get(key)
            if payload is not None:return payload
        return super().get(key)

    def put(self,key,payload,ownership=None):
        with self.buffer_guard:
            if ownership is not None:
                if key in self.owners and self.owners[key] is not ownership:
                    raise RuntimeError('Duplicate local chunk ownership')
                self.owners[key]=ownership
            previous=self.pending.get(key)
            self.pending[key]=payload
            self.pending_bytes+=len(payload)-(len(previous) if previous is not None else 0)
            # Bound open ownership descriptors even for tiny boundary chunks.
            if self.pending_bytes>=self.max_bytes or len(self.owners)>=256:self.flush()

    def flush(self):
        with self.buffer_guard:
            if not self.pending:return
            # Base put_many must inspect durable storage, not our pending map.
            pending=self.pending;self.pending={}
            try:super().put_many(pending)
            except BaseException:
                self.pending=pending
                raise
            self.pending_bytes=0
            for key in pending:
                owner=self.owners.pop(key,None)
                if owner is not None:owner.close()

class PackedCrops(SharedCrops):
    @contextmanager
    def _chunk_ownership(self,key):
        pack=self._packed_cache()
        stripe=zlib.crc32(f'{int(self.active_root)}:{key}'.encode())%16384
        lock=(self.disk_cache/'locks'/f'poll_{stripe}.lock').open('a')
        transferred=False
        try:
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    # Never wait for another owner while holding unpublished
                    # chunks: two workers could otherwise wait for each other.
                    pack.flush()
                    time.sleep(random.uniform(.01,.03))
            self.local.packed_ownership=(key,lock)
            self.local.packed_transferred=False
            try:
                yield lock
            finally:
                transferred=self.local.packed_transferred
                self.local.packed_ownership=None
        finally:
            if not transferred:lock.close()

    def _packed_cache(self):
        with self.chunk_lock:
            root=int(self.active_root)
            if getattr(self,'packed_root',None)!=root:
                self.packed=BufferedPackedCache(self.disk_cache/'cells'/str(root)/'packed',
                    max_bytes=int(os.environ.get('V1DD_PACKED_WRITE_BUFFER_MIB','64'))*1024**2)
                self.packed_root=root
            return self.packed

    def _read_cache(self,path,key):
        pack=self._packed_cache();payload=pack.get(key)
        if payload is not None:return payload
        if not path.exists():return None
        # Compatibility reads remain guarded while SQLite workers are active.
        # Migrate on first use so subsequent readers no longer need SQLite.
        payload=super()._read_cache(path,key)
        if payload is not None:pack.put(key,payload)
        return payload

    def _write_cache(self,path,key,payload):
        owner=getattr(self.local,'packed_ownership',None)
        if owner is None or owner[0]!=key:
            raise RuntimeError('Downloaded chunk must retain its ownership lock')
        self._packed_cache().put(key,payload,ownership=owner[1])
        self.local.packed_transferred=True

    def close_batch_connections(self):
        with self.chunk_lock:
            if self.chunk_pending:raise RuntimeError('Chunk readers must finish before flushing')
        if hasattr(self,'packed'):self.packed.flush()
        super().close_batch_connections()
        # Packs keep no open descriptors; their metadata can be reused by the
        # next batch of this root without retaining cache files after deletion.
