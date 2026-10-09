"""Cache and embed the axon-proofread V1DD ITC/DTC/neurogliaform cohort."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import fcntl
import json
import logging
import os
import threading
import time
import zlib
import hashlib
import shutil
import sqlite3
from v1dd_storage_budget import StorageBudget, projected_cell_bytes, prepared_bytes, tree_bytes
from v1dd_chunk_lifetimes import ChunkLifetimes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')
DB_ROOT = Path('/orcd/compute/sdorkenw/001/segclr-db')
EXPERIMENT = 'resnet_860b_reshuffled'
RUN = EXPERIMENT + '__20260603_150412'
CHECKPOINT = 'checkpoint_e0_s95000'
CHECKPOINT_PATH = Path('/orcd/data/sdorkenw/001/collina/segclr_runs/resnet_860b_reshuffled_20260603_150412/checkpoints/checkpoint_e0_s95000.pt')
MAT = 1196


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def cave_client():
    from caveclient import CAVEclient
    return CAVEclient('v1dd_public', server_address='https://global.em.brain.allentech.org', version=MAT)


def prepare():
    import pyarrow as pa
    from segclr_db import store as st, SegCLRWriter
    from segclr_db.skeletons import SkeletonCache, normalize_cave_skeleton
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE/'embeddings').mkdir(exist_ok=True)
    frame = pd.read_parquet(REPO/'analysis/v1dd_cell_types/fine_with_proofreading.parquet')
    frame = frame[frame.status_axon.eq(True) & frame.cell_type.str.startswith(('ITC', 'DTC', 'STC-Neurogliaform-'))].copy()
    assert not frame.pt_root_id.duplicated().any()
    frame['training_label'] = np.select([frame.cell_type.str.startswith('ITC'), frame.cell_type.str.startswith('DTC')], ['BipFam', 'MartFam'], default='NglFam')
    frame = frame.sort_values('pt_root_id')
    frame.to_parquet(BASE/'cohort.parquet', index=False)
    frame[['pt_root_id','target_id','cell_type','training_label','status_axon','strategy_axon']].to_csv(BASE/'cohort.csv', index=False)
    store = st.init_store(DB_ROOT, 'v1dd', datastack='v1dd_public', mat_version=MAT)
    writer = SegCLRWriter(store=store)
    source = st.open_store(DB_ROOT, 'microns')
    spec = st.scan(source, 'experiments', filter=f"experiment_id = '{EXPERIMENT}'").to_pylist()[0]
    writer.register_experiment(EXPERIMENT, json.loads(spec['spec_json']), embedding_dim=64,
                               config_yaml=spec['config_yaml'], spec_version=spec['spec_version'])
    run = writer.register_run(EXPERIMENT, CHECKPOINT_PATH.parent.parent, '20260603_150412')
    writer.register_checkpoints(run.run_id, [(CHECKPOINT, 0, 95000, str(CHECKPOINT_PATH))])
    stamp = pd.Timestamp.now('UTC').tz_localize(None)
    cells = []
    labels = []
    for row in frame.itertuples():
        rid = int(row.pt_root_id)
        cells.append(dict(root_id=rid,datastack='v1dd_public',mat_version=MAT,
            nucleus_id=int(row.target_id),is_neuron=True,proofread_axon=row.strategy_axon,
            proofread_dendrite=row.strategy_dendrite,
            soma_x_nm=float(row.pt_position_x),soma_y_nm=float(row.pt_position_y),soma_z_nm=float(row.pt_position_z),
            source_tables_json=json.dumps(['cell_type_multifeature_v1_fine','proofreading_status_and_strategy']),refreshed_ts=stamp))
        for label_set,label in [('cell_type',row.training_label),('v1dd_fine',row.cell_type)]:
            labels.append(dict(root_id=rid,label_set=label_set,label=label,source_table='cell_type_multifeature_v1_fine',mat_version=MAT,refreshed_ts=stamp))
    st.replace(store,'cells',pa.Table.from_pylist(cells),predicate='true')
    st.replace(store,'cell_labels',pa.Table.from_pylist(labels),predicate='true')
    client = cave_client()
    manifest = dict(dataset='v1dd',datastack='v1dd_public',materialization=MAT,
        materialization_metadata=client.materialize.get_version_metadata(MAT),
        counts=frame.training_label.value_counts().to_dict(),n_cells=len(frame),
        selection='status_axon=true; all ITC, all DTC, STC-Neurogliaform-',
        mapping='ITC -> putative BipFam; DTC -> putative MartFam; STC-Neurogliaform- -> NglFam',
        nodes='all CAVE v4 skeleton nodes',checkpoint=str(CHECKPOINT_PATH),run_id=RUN,checkpoint_id=CHECKPOINT,
        precision='fp16',compiled=True,compile_mode='reduce-overhead',batch_size=128,dynamic=False,
        compilation_cache='/orcd/scratch/orcd/013/jcbliao/torchinductor_cache/segclr_static_b128_fp16',
        crop_resolution_nm=[38.8,38.8,45],crop_size_voxels=129,
        image_source=client.info.image_source(),segmentation_source=client.info.segmentation_source())
    atomic_json(BASE/'manifest.json',manifest)
    cache = SkeletonCache(store)
    failures = []
    for i,rid in enumerate(frame.pt_root_id):
        rid = int(rid)
        try:
            if not cache.has_skeleton(rid):
                raw = client.skeleton.get_skeleton(rid,skeleton_version=4,output_format='dict')
                cache.store_skeletons([normalize_cave_skeleton(rid,raw,4)])
            print(f'CACHED {i+1}/{len(frame)} {rid}',flush=True)
        except Exception as exc:
            logging.exception('Skeleton failed %s',rid)
            failures.append(dict(root_id=rid,error=str(exc)))
    atomic_json(BASE/'prepare_status.json',dict(n_cells=len(frame),failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)} skeletons failed')


class V1DDCrops:
    """Remote volumes at matching resolution, with roots pinned to v1196."""
    box_size = 129

    def __init__(self, manifest):
        from cloudvolume import CloudVolume
        self.client = cave_client()
        self.timestamp = self.client.materialize.get_timestamp(MAT)
        self.local = threading.local()
        self.leaves = {}
        self.chunk_cache = OrderedDict()
        self.chunk_pending = {}
        self.batch_futures = None
        self.chunk_lock = threading.Lock()
        self.chunk_bytes = 0
        self.cache_connections = []
        self.chunk_budget = 2*1024**3
        self.chunk_pool = ThreadPoolExecutor(max_workers=16)
        self.disk_cache = BASE/'chunks_v1196_v1'
        (self.disk_cache/'locks').mkdir(parents=True,exist_ok=True)
        self.image_source = manifest['image_source']
        self.seg_source = manifest['segmentation_source']
        em = CloudVolume(self.image_source,use_https=True,progress=False,fill_missing=False)
        seg = CloudVolume(self.seg_source,use_https=True,progress=False,fill_missing=False,secrets=self.client.auth.token)
        self.em_mip = min(range(len(em.scales)),key=lambda i:abs(np.mean(em.scales[i]['resolution'][:2])-32))
        self.resolution = np.asarray(em.scales[self.em_mip]['resolution'],float)
        choices = [i for i,s in enumerate(seg.scales) if np.allclose(s['resolution'],self.resolution)]
        if not choices:
            raise ValueError('No segmentation scale matching EM resolution')
        self.seg_mip = choices[0]
        self.em_info = em.info
        self.seg_info = seg.info
        print(f'V1DD volumes: EM mip {self.em_mip}, segmentation mip {self.seg_mip}, resolution {self.resolution}',flush=True)

    def nm_to_voxel(self, xyz):
        return (np.asarray(xyz)/self.resolution).astype(np.int64)

    def _volumes(self):
        from cloudvolume import CloudVolume
        if not hasattr(self.local,'em'):
            self.local.em = CloudVolume(self.image_source,mip=self.em_mip,info=self.em_info,use_https=True,progress=False,fill_missing=False,bounded=False,cache=False,lru_bytes=0)
            self.local.seg = CloudVolume(self.seg_source,mip=self.seg_mip,info=self.seg_info,use_https=True,progress=False,fill_missing=False,bounded=False,secrets=self.client.auth.token,cache=False,lru_bytes=0)
        return self.local.em,self.local.seg

    def prepare_root(self, root_id):
        self.active_root = int(root_id)
        if root_id in self.leaves:
            return
        directory = BASE/'leaves'
        directory.mkdir(exist_ok=True)
        path = directory/f'{root_id}.npy'
        if path.exists():
            leaves = np.load(path)
        else:
            leaves = np.unique(self.client.chunkedgraph.get_leaves(root_id,stop_layer=1)).astype(np.uint64)
            temp = path.with_suffix(f'.{os.getpid()}.tmp.npy')
            np.save(temp,leaves)
            temp.replace(path)
        self.leaves = {root_id: np.sort(leaves.astype(np.uint64))}

    def _block(self, volume, center, segmentation=False):
        from cloudvolume.lib import Bbox
        lo = np.asarray(center,dtype=np.int64)-64
        hi = lo+129
        start = np.maximum(lo,volume.bounds.minpt)
        end = np.minimum(hi,volume.bounds.maxpt)
        out = np.zeros((129,129,129),dtype=volume.dtype)
        if np.any(start>=end):
            return out
        data = volume.download(Bbox(start,end),agglomerate=False) if segmentation else volume.download(Bbox(start,end))
        data = np.asarray(data)
        if data.ndim==4:
            data = data[...,0]
        dst = start-lo
        size = end-start
        out[dst[0]:dst[0]+size[0],dst[1]:dst[1]+size[1],dst[2]:dst[2]+size[2]] = data
        return out

    def fetch_crop(self, center, root_id):
        import torch
        em,seg = self._volumes()
        image = self._cached_block(em,center)
        mask = self._cached_block(seg,center,root_id)
        if not mask.any():
            raise ValueError(f'Empty v1196 root mask for {root_id}')
        return torch.from_numpy(image.transpose(2,1,0).copy())[None],torch.from_numpy(mask.transpose(2,1,0).copy())[None]

    def _mask(self,sv,root_id):
        ids = np.unique(sv)
        ids = ids[ids!=0]
        leaves = self.leaves[root_id]
        positions = np.searchsorted(leaves,ids)
        valid = positions<len(leaves)
        matching = np.zeros(len(ids),dtype=bool)
        matching[valid] = leaves[positions[valid]]==ids[valid]
        wanted = ids[matching]
        return np.isin(sv,wanted)

    def _chunk_future(self,volume,start,end,root_id):
        key = (root_id,tuple(start),tuple(end))
        with self.chunk_lock:
            if self.batch_futures is not None and key in self.batch_futures:
                return self.batch_futures[key]
            if key in self.chunk_cache:
                self.chunk_cache.move_to_end(key)
                future = Future()
                future.set_result(self.chunk_cache[key])
                if self.batch_futures is not None: self.batch_futures[key] = future
                return future
            future = self.chunk_pending.get(key)
            owner = future is None
            if owner:
                future = Future()
                self.chunk_pending[key] = future
            if self.batch_futures is not None: self.batch_futures[key] = future
        if owner:
            self.chunk_pool.submit(self._load_chunk,start,end,root_id,key,future)
        return future

    def _load_chunk(self,start,end,root_id,key,future):
        from cloudvolume.lib import Bbox
        try:
            em,seg = self._volumes()
            volume = seg if root_id is not None else em
            data = self._disk_chunk(volume,start,end,root_id)
            # Cache decoded image chunks and binary cell masks. Concurrent readers
            # share a single download/decode/mask operation for each chunk.
            data.setflags(write=False)
            with self.chunk_lock:
                self.chunk_cache[key] = data
                self.chunk_bytes += data.nbytes
                while self.chunk_bytes>self.chunk_budget and len(self.chunk_cache)>1:
                    _,old = self.chunk_cache.popitem(last=False)
                    self.chunk_bytes -= old.nbytes
                self.chunk_pending.pop(key)
                future.set_result(data)
        except BaseException as exc:
            with self.chunk_lock:
                self.chunk_pending.pop(key,None)
                future.set_exception(exc)

    def _cache_connection(self):
        rid = int(self.active_root)
        connections = getattr(self.local,'cache_connections',None)
        if connections is None:
            connections = self.local.cache_connections = {}
        connection = connections.get(rid)
        if connection is None:
            folder = self.disk_cache/'cells'/str(rid)
            folder.mkdir(parents=True,exist_ok=True)
            connection = sqlite3.connect(folder/'chunks.sqlite',timeout=120,check_same_thread=False)
            connection.execute('PRAGMA synchronous=NORMAL')
            if not (folder/'chunks.sqlite').stat().st_size:
                connection.execute('PRAGMA auto_vacuum=INCREMENTAL')
            connection.execute('CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, data BLOB NOT NULL)')
            connection.commit()
            connections[rid] = connection
            with self.chunk_lock:
                self.cache_connections.append(connection)
        return connection

    def _disk_chunk(self,volume,start,end,root_id):
        from cloudvolume.lib import Bbox
        shape = tuple(map(int,np.asarray(end)-start))
        name = '_'.join(map(str,tuple(start)+tuple(end)))
        rid = int(self.active_root)
        key = ('image:' if root_id is None else 'mask:')+name
        connection = self._cache_connection()
        def read():
            row = connection.execute('SELECT data FROM chunks WHERE key=?',(key,)).fetchone()
            if row is None: return None
            if root_id is not None:
                return np.unpackbits(np.frombuffer(row[0],np.uint8),count=int(np.prod(shape))).reshape(shape).view(bool)
            return np.frombuffer(row[0],dtype=volume.dtype).reshape(shape)
        cached = read()
        if cached is not None: return cached
        lock_id = zlib.crc32(f'{rid}:{key}'.encode())%4096
        with (self.disk_cache/'locks'/f'{lock_id}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            cached = read()
            if cached is not None: return cached
            # Reuse existing cell-specific masks during the transition.
            legacy = self.disk_cache/'mask_mip2'/str(rid)/str(zlib.crc32(name.encode())%256)/f'{name}.npy'
            if root_id is not None and legacy.exists():
                data = np.unpackbits(np.load(legacy,allow_pickle=False),count=int(np.prod(shape))).reshape(shape).view(bool)
            else:
                box = Bbox(start,end)
                from cloudvolume.exceptions import EmptyVolumeException
                try:
                    data = volume.download(box,agglomerate=False) if root_id is not None else volume.download(box)
                except EmptyVolumeException:
                    if root_id is None: raise
                    # Sparse segmentation can omit background-only storage chunks.
                    # The assembled crop must still contain the target cell mask.
                    logging.warning('Missing segmentation chunk treated as background root=%s bbox=%s',rid,box)
                    data = np.zeros(shape,dtype=volume.dtype)
                data = np.asarray(data)
                if data.ndim==4: data = data[...,0]
                if root_id is not None: data = self._mask(data,root_id)
            payload = np.packbits(data).tobytes() if root_id is not None else data.tobytes()
            connection.execute('INSERT OR IGNORE INTO chunks VALUES (?,?)',(key,payload))
            connection.commit()
            return data

    def configure_reclamation(self, plan):
        self.lifetimes = plan
        connection = self._cache_connection()
        if connection.execute('PRAGMA auto_vacuum').fetchone()[0]!=2:
            print(f'CACHE MIGRATION root={self.active_root} enabling incremental vacuum',flush=True)
            connection.execute('PRAGMA auto_vacuum=INCREMENTAL')
            connection.execute('VACUUM')
            print(f'CACHE MIGRATED root={self.active_root}',flush=True)

    def prefetch_batch(self,batch_index):
        """Schedule only this batch's exact chunks together, with shared futures."""
        em,seg = self._volumes()
        with self.chunk_lock:
            self.batch_futures = {}
        for key in sorted(self.lifetimes.batch_keys.get(batch_index,[]),key=lambda k:zlib.crc32(k.encode())):
            packed,start,end = self.lifetimes.items[key]
            self._chunk_future(seg if packed else em,np.asarray(start),np.asarray(end),self.active_root if packed else None)

    def expire_batch(self, batch_index):
        """Delete chunks only after all their dependent input batches are durable."""
        keys = self.lifetimes.expired.get(batch_index,[])
        with self.chunk_lock:
            if self.chunk_pending:
                raise RuntimeError('Cannot expire chunks while readers are active')
            self.batch_futures = None
        if not keys: return
        with self.chunk_lock:
            for key in keys:
                packed,start,end = self.lifetimes.items[key]
                cache_key = (self.active_root if packed else None,start,end)
                data = self.chunk_cache.pop(cache_key,None)
                if data is not None: self.chunk_bytes -= data.nbytes
        connection = self._cache_connection()
        path = self.disk_cache/'cells'/str(self.active_root)/'chunks.sqlite'
        before = path.stat().st_size
        # Reclaim the entire free list in this single drained-batch transaction.
        # This Python/SQLite build reclaims just one page per incremental PRAGMA.
        connection.execute('PRAGMA auto_vacuum=FULL')
        connection.executemany('DELETE FROM chunks WHERE key=?',[(key,) for key in keys])
        connection.commit()
        connection.execute('PRAGMA auto_vacuum=INCREMENTAL')
        for key in keys:
            if key.startswith('mask:'):
                name = key[5:]
                legacy = self.disk_cache/'mask_mip2'/str(self.active_root)/str(zlib.crc32(name.encode())%256)/f'{name}.npy'
                legacy.unlink(missing_ok=True)
        after = path.stat().st_size
        print(f'CHUNKS EXPIRED root={self.active_root} batch={batch_index} chunks={len(keys)} sqlite_reclaimed_mib={(before-after)/1024**2:.2f}',flush=True)

    def release_root(self, root_id):
        """All crop futures must be drained before releasing this cell's chunks."""
        with self.chunk_lock:
            if self.chunk_pending:
                raise RuntimeError('Cannot release chunks while readers are active')
            self.chunk_cache.clear()
            self.chunk_bytes = 0
            connections = self.cache_connections
            self.cache_connections = []
        for connection in connections:
            connection.close()
        # Connections in chunk-reader thread locals are retired with their root;
        # subsequent roots use different dictionary keys.
        shutil.rmtree(self.disk_cache/'cells'/str(root_id),ignore_errors=True)
        shutil.rmtree(self.disk_cache/'mask_mip2'/str(root_id),ignore_errors=True)
        if (self.disk_cache/'cells'/str(root_id)).exists() or (self.disk_cache/'mask_mip2'/str(root_id)).exists():
            raise RuntimeError(f'Chunk cleanup incomplete for {root_id}')
        print(f'CELL CHUNKS RELEASED {root_id}',flush=True)

    def chunk_boxes(self,volume,center):
        lo = np.asarray(center,dtype=np.int64)-64
        start = np.maximum(lo,volume.bounds.minpt)
        end = np.minimum(lo+129,volume.bounds.maxpt)
        if np.any(start>=end):
            return
        size = np.asarray(volume.chunk_size,dtype=np.int64)
        offset = np.asarray(volume.voxel_offset,dtype=np.int64)
        first = ((start-offset)//size)*size+offset
        for x in range(int(first[0]),int(end[0]),int(size[0])):
            for y in range(int(first[1]),int(end[1]),int(size[1])):
                for z in range(int(first[2]),int(end[2]),int(size[2])):
                    yield (np.maximum([x,y,z],volume.bounds.minpt),
                           np.minimum(np.asarray([x,y,z])+size,volume.bounds.maxpt))

    def _cached_block(self,volume,center,root_id=None):
        lo = np.asarray(center,dtype=np.int64)-64
        hi = lo+129
        start = np.maximum(lo,volume.bounds.minpt)
        end = np.minimum(hi,volume.bounds.maxpt)
        out = np.zeros((129,129,129),dtype=bool if root_id is not None else volume.dtype)
        if np.any(start>=end):
            return out
        blocks = []
        for chunk_start,chunk_end in self.chunk_boxes(volume,center):
            future = self._chunk_future(volume,chunk_start,chunk_end,root_id)
            left = np.maximum(start,chunk_start)
            right = np.minimum(end,chunk_end)
            dst = tuple(slice(int(a),int(b)) for a,b in zip(left-lo,right-lo))
            src = tuple(slice(int(a),int(b)) for a,b in zip(left-chunk_start,right-chunk_start))
            blocks.append((future,dst,src))
        for future,dst,src in blocks:
            out[dst] = future.result()[src]
        return out


INPUTS = BASE/'inputs_v1'


def make_inference(manifest,prepared=False):
    import torch
    from segclr.inference import SegCLRInference
    from segclr.data.crops import mask_and_normalize
    from segclr.utils.checkpointing import load_checkpoint_automatic

    class Inference(SegCLRInference):
        def __init__(self):
            self.batch_size = 128
            self.num_threads = int(os.environ.get('V1DD_CROP_THREADS','16'))
            self.precision = 'fp16'
            self.channels_last = True
            self.allow_partial = False
            self.device = torch.device('cuda')
            self._apply_precision()
            self.crops = (SimpleNamespace(box_size=129,resolution=np.array(manifest.get('crop_resolution_nm',[38.8,38.8,45])))
                          if prepared else V1DDCrops(manifest))
            self.model = load_checkpoint_automatic(CHECKPOINT_PATH,self.device).eval().to(memory_format=torch.channels_last_3d)
            assert self.model.bottleneck_dim==64
            self.model = torch.compile(self.model,mode='reduce-overhead',dynamic=False)
            self.experiment = SimpleNamespace(embedding_dim=64)
            self._buffers = self._buffer_views = None
            self._n_crop_failures = 0

        def _embed(self,root_id,skeleton,node_ids):
            if prepared:
                folder = INPUTS/str(root_id)
                ready = json.loads((folder/'ready.json').read_text())
                assert ready['n_nodes']==len(skeleton) and ready['materialization']==MAT
                assert ready['coords_sha256']==hashlib.sha256(skeleton.coords.tobytes()).hexdigest()
                ids,vectors = self._run_prepared(root_id,[folder/f'batch_{i:06d}' for i in range(ready['n_batches'])])
                order = np.argsort(ids)
                np.testing.assert_array_equal(ids[order],node_ids)
                return ids[order],vectors[order]
            self.crops.prepare_root(root_id)
            self._progress_nodes = 0
            return super()._embed(root_id,skeleton,node_ids)

        def _await_fill(self,root_id,futures):
            started = time.monotonic()
            result = super()._await_fill(root_id,futures)
            self._progress_nodes = getattr(self,'_progress_nodes',0)+len(result[0])
            print(f'BATCH root={root_id} nodes={self._progress_nodes} wait_s={time.monotonic()-started:.3f}',flush=True)
            return result

        def _forward(self,slot,n_ok):
            started = time.monotonic()
            em,mask = self._buffers[slot]
            # Every model invocation has exactly 128 rows, including the final batch.
            if n_ok<128:
                em[n_ok:].zero_()
                mask[n_ok:].zero_()
            batch = mask_and_normalize(em.to(self.device,non_blocking=True),mask.to(self.device,non_blocking=True))
            batch = batch.to(memory_format=torch.channels_last_3d)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                _,vectors = self.model(batch,return_embeddings=True)
            result = vectors[:n_ok].float().cpu().numpy()
            if result.shape!=(n_ok,64) or not np.isfinite(result).all():
                raise ValueError('Invalid embeddings')
            print(f'FORWARD nodes={n_ok} seconds={time.monotonic()-started:.3f}',flush=True)
            return result

        def _run_prepared(self,root_id,paths):
            self._ensure_buffers()
            def read(path,slot):
                ids = np.load(path/'node_ids.npy',allow_pickle=False)
                image = np.load(path/'image.npy',mmap_mode='r',allow_pickle=False)
                packed = np.load(path/'mask.npy',allow_pickle=False)
                n = len(ids)
                assert 0<n<=128 and image.shape==(n,1,129,129,129) and image.dtype==np.uint8
                assert packed.shape==(n,(129**3+7)//8) and packed.dtype==np.uint8
                em_view,mask_view = self._buffer_views[slot]
                em_view[:n] = image
                mask_view[:n] = np.unpackbits(packed,axis=1,count=129**3).reshape(n,1,129,129,129)
                return ids
            out_ids,out_vectors = [],[]
            with ThreadPoolExecutor(max_workers=1) as reader:
                pending = reader.submit(read,paths[0],0)
                for index,path in enumerate(paths):
                    slot = index%2
                    current = pending
                    pending = reader.submit(read,paths[index+1],1-slot) if index+1<len(paths) else None
                    started = time.monotonic()
                    ids = current.result()
                    wait = time.monotonic()-started
                    vectors = self._forward(slot,len(ids))
                    out_ids.extend(ids)
                    out_vectors.extend(vectors)
                    print(f'PREPARED root={root_id} batch={path.name} nodes={len(ids)} input_wait_s={wait:.3f}',flush=True)
            return np.asarray(out_ids),np.asarray(out_vectors)
    return Inference()


def prefetch_order(centers):
    # NumPy-only equivalent of SegCLR's morton_order, avoiding torch on CPU jobs.
    def spread(values):
        v = values.astype(np.int64) & 0x1FFFFF
        for shift,mask in [(32,0x1F00000000FFFF),(16,0x1F0000FF0000FF),
                           (8,0x100F00F00F00F00F),(4,0x10C30C30C30C30C3),
                           (2,0x1249249249249249)]:
            v = (v | (v << shift)) & mask
        return v
    cells = ((centers-centers.min(axis=0))//64).astype(np.int64)
    codes = spread(cells[:,0]) | (spread(cells[:,1])<<1) | (spread(cells[:,2])<<2)
    return np.argsort(codes,kind='stable')


def prefetch(args):
    """Populate the shared chunk cache without loading torch or a model."""
    from segclr_db import store as st
    from segclr_db.skeletons import SkeletonCache
    manifest = json.loads((BASE/'manifest.json').read_text())
    crops = V1DDCrops(manifest)
    cache = SkeletonCache(st.open_store(DB_ROOT,'v1dd'))
    frame = pd.read_parquet(BASE/'cohort.parquet')
    deadline = time.monotonic()+args.max_seconds
    completed = []
    n_chunks = 0
    try:
        for rid in frame.pt_root_id.to_numpy()[args.rank::args.world]:
            rid = int(rid)
            if (BASE/'embeddings'/f'{rid}.npz').exists():
                continue
            if time.monotonic()>=deadline:
                break
            crops.prepare_root(rid)
            skeleton = cache.get_skeleton(rid,fetch_if_missing=False)
            centers = crops.nm_to_voxel(skeleton.coords)
            # Same spatial order as inference, to prepare its next batches first.
            centers = centers[prefetch_order(centers)]
            em,seg = crops._volumes()
            seen = set()
            pending = []
            stopped = False
            for center in centers:
                for volume,root in [(em,None),(seg,rid)]:
                    for start,end in crops.chunk_boxes(volume,center):
                        key = (root,tuple(start),tuple(end))
                        if key in seen:
                            continue
                        seen.add(key)
                        pending.append(crops._chunk_future(volume,start,end,root))
                        if len(pending)>=256:
                            for future in pending:
                                future.result()
                            n_chunks += len(pending)
                            pending.clear()
                            print(f'PREFETCH root={rid} chunks={len(seen)} total={n_chunks}',flush=True)
                            if time.monotonic()>=deadline:
                                stopped = True
                                break
                    if stopped:
                        break
                if stopped:
                    break
            for future in pending:
                future.result()
            n_chunks += len(pending)
            if stopped:
                break
            completed.append(rid)
            print(f'CACHED ROOT {rid} chunks={len(seen)}',flush=True)
    finally:
        crops.chunk_pool.shutdown(wait=True)
    job_id = os.environ.get('SLURM_ARRAY_JOB_ID','local')
    atomic_json(BASE/f'prefetch_status_{job_id}_{args.rank}.json',
                dict(completed_roots=completed,n_chunks=n_chunks,cache=str(crops.disk_cache)))


def prepare_inputs(args):
    """Build complete fixed-size inference batches using CPU-only workers."""
    from segclr_db import store as st
    from segclr_db.skeletons import SkeletonCache
    manifest = json.loads((BASE/'manifest.json').read_text())
    crops = V1DDCrops(manifest)
    cache = SkeletonCache(st.open_store(DB_ROOT,'v1dd'))
    rank,world = args.rank,args.world
    assignment_path = BASE/'cpu_assignment.json'
    if assignment_path.exists() and not args.input_root:
        assignment = json.loads(assignment_path.read_text())
        job = os.environ.get('SLURM_ARRAY_JOB_ID','local')
        if job in assignment['offsets']:
            rank += assignment['offsets'][job]
            world = assignment['world']
    roots = ([args.input_root] if args.input_root else
             pd.read_parquet(BASE/'cohort.parquet').pt_root_id.to_numpy()[rank::world])
    print(f'CPU ASSIGNMENT rank={rank} world={world}',flush=True)
    (BASE/'preparation_cell_locks').mkdir(exist_ok=True)
    deadline = time.monotonic()+args.max_seconds
    built = 0
    budget = StorageBudget(BASE) if args.storage_budget_gib else None
    try:
        with ThreadPoolExecutor(max_workers=16) as readers:
            for root in roots:
                rid = int(root)
                cell_lock = (BASE/'preparation_cell_locks'/f'{rid}.lock').open('a')
                try:
                    fcntl.flock(cell_lock,fcntl.LOCK_EX)
                    if not args.input_root and (BASE/'embeddings'/f'{rid}.npz').exists():
                        crops.release_root(rid)
                        if budget: budget.release(rid)
                        continue
                    folder = INPUTS/str(rid)
                    if (folder/'ready.json').exists() and not args.input_limit_batches:
                        crops.release_root(rid)
                        if budget:
                            ready = json.loads((folder/'ready.json').read_text())
                            budget.ready(rid,prepared_bytes(ready['n_nodes']))
                        if args.cell_lifecycle and not budget:
                            while folder.exists():
                                if time.monotonic()>deadline: return False
                                time.sleep(10)
                        continue
                    folder.mkdir(parents=True,exist_ok=True)
                    crops.prepare_root(rid)
                    skeleton = cache.get_skeleton(rid,fetch_if_missing=False)
                    centers = crops.nm_to_voxel(skeleton.coords)
                    order = prefetch_order(centers)
                    lifetimes = ChunkLifetimes(crops,centers,order)
                    if budget:
                        reservation = lifetimes.peak_total_bytes
                        # Existing databases may contain chunks prefetched ahead of
                        # their first use; include them and one-time VACUUM workspace.
                        reservation += 2*tree_bytes(crops.disk_cache/'cells'/str(rid))
                        # Legacy masks coexist with SQLite copies until cell cleanup.
                        reservation += tree_bytes(crops.disk_cache/'mask_mip2'/str(rid))
                        reservation += sum(tree_bytes(p) for p in folder.glob('.batch_*.tmp') if p.is_dir())
                        reported = False
                        while not budget.reserve(rid,reservation):
                            if not reported:
                                print(f'STORAGE BUDGET WAIT root={rid} reserve_gib={reservation/1024**3:.2f}',flush=True)
                                reported = True
                            if time.monotonic()>deadline: return False
                            time.sleep(10)
                        print(f'STORAGE RESERVED root={rid} gib={reservation/1024**3:.2f}',flush=True)
                    crops.configure_reclamation(lifetimes)
                    print(f'CHUNK PLAN root={rid} peak_chunks_gib={lifetimes.peak_chunk_bytes/1024**3:.2f} peak_total_gib={lifetimes.peak_total_bytes/1024**3:.2f} full_chunks_gib={lifetimes.full_cache_bytes/1024**3:.2f}',flush=True)
                    coords_hash = hashlib.sha256(skeleton.coords.tobytes()).hexdigest()
                    n_batches = (len(order)+127)//128
                    for batch_index in range(n_batches):
                        if time.monotonic()>deadline:
                            print('INPUT PREPARATION CONTINUES: time budget reached; complete batches are saved',flush=True)
                            return False
                        target = folder/f'batch_{batch_index:06d}'
                        ids = order[batch_index*128:(batch_index+1)*128]
                        with (folder/f'batch_{batch_index:06d}.lock').open('a') as lock:
                            fcntl.flock(lock,fcntl.LOCK_EX)
                            if target.exists():
                                np.testing.assert_array_equal(np.load(target/'node_ids.npy'),ids)
                                assert json.loads((target/'metadata.json').read_text())['coords_sha256']==coords_hash
                            else:
                                started = time.monotonic()
                                crops.prefetch_batch(batch_index)
                                image = np.empty((len(ids),1,129,129,129),np.uint8)
                                mask = np.empty((len(ids),(129**3+7)//8),np.uint8)
                                def read(item):
                                    row,node = item
                                    em,seg = crops._volumes()
                                    xyz = centers[node]
                                    im = crops._cached_block(em,xyz)
                                    mk = crops._cached_block(seg,xyz,rid)
                                    if not mk.any():
                                        raise ValueError(f'Empty mask: root {rid}, node {node}')
                                    image[row,0] = im.transpose(2,1,0)
                                    mask[row] = np.packbits(mk.transpose(2,1,0).reshape(-1))
                                for _ in readers.map(read,enumerate(ids)):
                                    pass
                                temp = folder/f'.batch_{batch_index:06d}.{os.getpid()}.tmp'
                                temp.mkdir(exist_ok=True)
                                np.save(temp/'image.npy',image,allow_pickle=False)
                                np.save(temp/'mask.npy',mask,allow_pickle=False)
                                np.save(temp/'node_ids.npy',ids,allow_pickle=False)
                                atomic_json(temp/'metadata.json',dict(root_id=rid,materialization=MAT,
                                    coords_sha256=coords_hash,n_nodes=len(ids),crop_size=129,batch_size=128))
                                temp.replace(target)
                                print(f'INPUT READY root={rid} batch={batch_index} nodes={len(ids)} seconds={time.monotonic()-started:.3f}',flush=True)
                        crops.expire_batch(batch_index)
                        built += 1
                        if args.input_limit_batches and built>=args.input_limit_batches:
                            return
                    atomic_json(folder/'ready.json',dict(root_id=rid,materialization=MAT,
                        n_nodes=len(order),n_batches=n_batches,coords_sha256=coords_hash))
                    print(f'INPUT ROOT READY {rid} nodes={len(order)} batches={n_batches}',flush=True)
                    crops.release_root(rid)
                    if budget: budget.ready(rid,prepared_bytes(len(order)))
                    if args.cell_lifecycle and not budget:
                        # Keep at most one unconsumed complete cell per CPU worker.
                        while folder.exists():
                            if time.monotonic()>deadline: return False
                            time.sleep(10)
                finally:
                    cell_lock.close()
    finally:
        crops.chunk_pool.shutdown(wait=True)


def benchmark_inputs(args):
    import torch
    manifest = json.loads((BASE/'manifest.json').read_text())
    engine = make_inference(manifest,prepared=True)
    paths = sorted(p for p in (INPUTS/str(args.input_root)).glob('batch_[0-9]*') if p.is_dir())[:args.input_limit_batches or 4]
    assert paths and all(len(np.load(path/'node_ids.npy'))==128 for path in paths)
    engine._run_prepared(args.input_root,paths[:1])  # Warm compilation cache.
    times = []
    for trial in range(3):
        torch.cuda.synchronize()
        started = time.monotonic()
        ids,vectors = engine._run_prepared(args.input_root,paths)
        seconds = time.monotonic()-started
        times.append(seconds)
        print(f'INPUT BENCHMARK trial={trial} embeddings={len(ids)} seconds={seconds:.3f} per_minute={len(ids)*60/seconds:.1f}',flush=True)
    atomic_json(BASE/'prepared_input_benchmark.json',dict(root_id=args.input_root,
        embeddings_per_trial=len(ids),seconds=times,embeddings_per_minute=[len(ids)*60/t for t in times],
        gpu=torch.cuda.get_device_name(),batch_size=128,precision='fp16',compiled=True))


def infer(args):
    from segclr_db import store as st, SegCLRWriter
    from segclr_db.skeletons import SkeletonCache
    import torch
    assert torch.cuda.is_available()
    manifest = json.loads((BASE/'manifest.json').read_text())
    store = st.open_store(DB_ROOT,'v1dd')
    cache = SkeletonCache(store)
    writer = SegCLRWriter(store=store)
    frame = pd.read_parquet(BASE/'cohort.parquet')
    roots = [int(x) for x in frame.pt_root_id][args.rank::args.world]
    if args.shared_queue:
        roots = [int(x) for x in frame.pt_root_id]
    if args.input_root:
        roots = [int(args.input_root)]
    if args.pilot:
        roots = roots[:1]
    engine = None
    failures = []
    deadline = time.monotonic()+args.max_seconds
    pending = list(roots)
    budget = StorageBudget(BASE) if args.storage_budget_gib else None
    if args.shared_queue:
        (BASE/'embedding_commits').mkdir(exist_ok=True)
        (BASE/'inference_cell_locks').mkdir(exist_ok=True)
    while pending:
        if time.monotonic()>deadline:
            print('INFERENCE CONTINUES: time budget reached',flush=True)
            return False
        cell_lock = None
        if args.shared_queue:
            pending = [r for r in pending if not (BASE/'embedding_commits'/f'{r}.json').exists()]
            if not pending: break
            rid = None
            for candidate in pending:
                if not ((BASE/'embeddings'/f'{candidate}.npz').exists() or (INPUTS/str(candidate)/'ready.json').exists()): continue
                handle = (BASE/'inference_cell_locks'/f'{candidate}.lock').open('a')
                try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                if (BASE/'embedding_commits'/f'{candidate}.json').exists():
                    handle.close()
                    continue
                rid,cell_lock = candidate,handle
                break
        else:
            rid = next((r for r in pending if (BASE/'embeddings'/f'{r}.npz').exists()
                        or not args.cell_lifecycle or (INPUTS/str(r)/'ready.json').exists()),None)
        if rid is None:
            time.sleep(10)
            continue
        pending.remove(rid)
        try:
            skeleton = cache.get_skeleton(rid,fetch_if_missing=False)
            path = BASE/'embeddings'/f'{rid}.npz'
            if args.pilot:
                engine = make_inference(manifest)
                nodes = np.unique(np.linspace(0,len(skeleton)-1,min(129,len(skeleton)),dtype=int))
                ids,vectors = engine._embed(rid,skeleton,nodes)
                np.testing.assert_array_equal(ids,nodes)
                atomic_json(BASE/'pilot_status.json',dict(root_id=rid,n_nodes=len(ids),status='passed',batch_size=128,precision='fp16',compiled=True,
                    gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,compile_cache=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
                    resolution_nm=engine.crops.resolution.tolist()))
                print('PILOT PASSED',rid,len(ids),flush=True)
                return
            if path.exists():
                with np.load(path) as saved:
                    ids,vectors = saved['node_ids'],saved['embeddings']
                    assert int(saved['root_id'])==rid and str(saved['checkpoint_id'])==CHECKPOINT
            else:
                if engine is None:
                    engine = make_inference(manifest,prepared=args.prepared_inputs)
                ids,vectors = engine._embed(rid,skeleton,np.arange(len(skeleton)))
                np.testing.assert_array_equal(ids,np.arange(len(skeleton)))
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,root_id=rid,node_ids=ids,embeddings=vectors,checkpoint_id=CHECKPOINT,run_id=RUN,materialization=MAT)
                tmp.replace(path)
            assert vectors.shape==(len(skeleton),64) and np.isfinite(vectors).all()
            np.testing.assert_array_equal(ids,np.arange(len(skeleton)))
            with (BASE/'commit.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                writer.add_cell_embeddings(EXPERIMENT,rid,ids,vectors,checkpoint_id=CHECKPOINT,task_id=args.rank)
            print('VERIFIED',rid,len(ids),flush=True)
            if args.cell_lifecycle:
                shutil.rmtree(INPUTS/str(rid),ignore_errors=True)
                if (INPUTS/str(rid)).exists():
                    raise RuntimeError(f'Input cleanup incomplete for {rid}')
                print('CELL INPUTS RELEASED',rid,flush=True)
                if budget: budget.release(rid)
            if args.shared_queue:
                atomic_json(BASE/'embedding_commits'/f'{rid}.json',dict(root_id=rid,n_nodes=len(ids),checkpoint_id=CHECKPOINT))
        except Exception as exc:
            logging.exception('Inference failed %s',rid)
            failures.append(dict(root_id=rid,error=str(exc)))
            if args.pilot:
                raise
        finally:
            if cell_lock is not None: cell_lock.close()
    status_name = f'inference_status_cell_{args.input_root}.json' if args.input_root else f'inference_status_{args.rank}.json'
    atomic_json(BASE/status_name,dict(n_cells=len(roots),failures=failures))
    if failures:
        raise RuntimeError(f'{len(failures)} cells failed')


def verify():
    from segclr_db import store as st
    from segclr_db.skeletons import SkeletonCache
    store = st.open_store(DB_ROOT,'v1dd')
    cache = SkeletonCache(store)
    frame = pd.read_parquet(BASE/'cohort.parquet')
    total = 0
    for rid in frame.pt_root_id:
        rid = int(rid)
        skeleton = cache.get_skeleton(rid,fetch_if_missing=False)
        with np.load(BASE/'embeddings'/f'{rid}.npz') as saved:
            assert int(saved['root_id'])==rid and str(saved['run_id'])==RUN
            assert str(saved['checkpoint_id'])==CHECKPOINT and int(saved['materialization'])==MAT
            np.testing.assert_array_equal(saved['node_ids'],np.arange(len(skeleton)))
            vectors = saved['embeddings']
            assert vectors.shape==(len(skeleton),64) and np.isfinite(vectors).all()
            got = st.scan(store,'node_embeddings',dim=64,filter=f"root_id = {rid} AND run_id = '{RUN}' AND checkpoint_id = '{CHECKPOINT}'").to_pandas().sort_values('node_id')
            np.testing.assert_array_equal(got.node_id.to_numpy(),np.arange(len(skeleton)))
            np.testing.assert_array_equal(np.stack(got.embedding),vectors)
        total += len(skeleton)
    summary = dict(status='verified',n_cells=len(frame),n_embeddings=total,counts=frame.training_label.value_counts().to_dict())
    atomic_json(BASE/'verified_summary.json',summary)
    print(summary,flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--verify',action='store_true')
    ap.add_argument('--prefetch',action='store_true')
    ap.add_argument('--prepare-inputs',action='store_true')
    ap.add_argument('--prepared-inputs',action='store_true')
    ap.add_argument('--cell-lifecycle',action='store_true')
    ap.add_argument('--shared-queue',action='store_true')
    ap.add_argument('--storage-budget-gib',type=float,default=0)
    ap.add_argument('--benchmark-inputs',action='store_true')
    ap.add_argument('--input-root',type=int)
    ap.add_argument('--input-limit-batches',type=int,default=0)
    ap.add_argument('--max-seconds',type=float,default=42000)
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--world',type=int,default=1)
    args = ap.parse_args()
    assert 0<=args.rank<args.world
    logging.basicConfig(level=logging.INFO)
    if args.benchmark_inputs:
        benchmark_inputs(args)
    elif args.prepare_inputs:
        if prepare_inputs(args) is False:
            raise SystemExit(75)
    elif args.prefetch:
        prefetch(args)
    elif args.prepare:
        prepare()
    elif args.verify:
        verify()
    else:
        if infer(args) is False:
            raise SystemExit(75)


if __name__=='__main__':
    main()
