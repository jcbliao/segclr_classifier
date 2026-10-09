"""Cooperative CPU workers for unfinished V1DD inference batches."""
import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,'/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot')
from segclr_db import store as st
from segclr_db.skeletons import SkeletonCache
from infer_v1dd_candidates import BASE,DB_ROOT,MAT,atomic_json,prefetch_order
from v1dd_batch_queue import BatchQueue
from v1dd_packed_chunks import PackedCrops
from v1dd_chunk_lifetimes import ChunkLifetimes
from v1dd_storage_budget import StorageBudget,prepared_bytes,tree_bytes,GIB


def initialize():
    cache=SkeletonCache(st.open_store(DB_ROOT,'v1dd'))
    roots=pd.read_parquet(BASE/'cohort.parquet').pt_root_id
    counts=[(int(r),len(cache.get_skeleton(int(r),fetch_if_missing=False))) for r in roots]
    # Old workers must be stopped before initialization/reconciliation.
    for folder in (BASE/'inputs_v1').iterdir():
        if folder.is_dir():
            for temp in folder.glob('.batch_*.tmp'):shutil.rmtree(temp)
    budget=StorageBudget(BASE);state=budget.initialize(600*GIB)
    # Reserve common headroom for queue metadata and compressed root plans.
    budget._change(lambda s:s.update(common_bytes=s['common_bytes']+2*GIB))
    BatchQueue(BASE).initialize(counts,window=None)
    print('BATCH QUEUE INITIALIZED',len(counts),'cells; existing GiB',
          (state['common_bytes']+sum(v['bytes'] for v in state['cells'].values()))/GIB,flush=True)


def load_plan(queue,rid):
    with gzip.open(queue.plans/f'{rid}.pkl.gz','rb') as handle:return pickle.load(handle)


def setup_plan(queue,crops,skeleton_cache,rid):
    path=queue.plans/f'{rid}.pkl.gz'
    if path.exists():payload=load_plan(queue,rid)
    else:
        skeleton=skeleton_cache.get_skeleton(rid,fetch_if_missing=False)
        centers=crops.nm_to_voxel(skeleton.coords);order=prefetch_order(centers)
        plan=ChunkLifetimes(crops,centers,order,window=len(order)//128+1)
        digest=hashlib.sha256(skeleton.coords.tobytes()).hexdigest()
        # Validate saved work once before putting its cell into the shared queue.
        folder=BASE/'inputs_v1'/str(rid)
        for i in range(plan.n_batches):
            target=folder/f'batch_{i:06d}'
            if target.exists():
                meta=json.loads((target/'metadata.json').read_text())
                assert meta['coords_sha256']==digest
                np.testing.assert_array_equal(np.load(target/'node_ids.npy'),order[i*128:(i+1)*128])
        payload=dict(plan=plan,centers=centers,order=order,coords_sha256=digest)
        temp=path.with_suffix(f'.{os.getpid()}.tmp')
        with gzip.open(temp,'wb',compresslevel=1) as handle:pickle.dump(payload,handle,protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
    reservation=payload['plan'].full_cache_bytes+prepared_bytes(len(payload['order']))
    reservation+=2*tree_bytes(BASE/'chunks_v1196_v1'/'cells'/str(rid))
    reservation+=tree_bytes(BASE/'chunks_v1196_v1'/'mask_mip2'/str(rid))
    queue.planned(rid,reservation)
    print(f'BATCH CELL PLANNED root={rid} reserve_gib={reservation/GIB:.2f} batches={payload["plan"].n_batches}',flush=True)
    return payload


def prepare_batch(queue,crops,payload,rid,index,readers):
    folder=BASE/'inputs_v1'/str(rid);folder.mkdir(exist_ok=True)
    target=folder/f'batch_{index:06d}';ids=payload['order'][index*128:(index+1)*128]
    if target.exists():
        np.testing.assert_array_equal(np.load(target/'node_ids.npy'),ids)
        assert json.loads((target/'metadata.json').read_text())['coords_sha256']==payload['coords_sha256']
        queue.complete(rid,index);return
    for temp in folder.glob(f'.batch_{index:06d}.*.tmp'):shutil.rmtree(temp)
    started=time.monotonic();crops.select_root(rid,payload['plan']);crops.prefetch_batch(index)
    image=np.empty((len(ids),1,129,129,129),np.uint8);mask=np.empty((len(ids),(129**3+7)//8),np.uint8)
    def read(item):
        row,node=item;em,seg=crops._volumes();xyz=payload['centers'][node]
        im=crops._cached_block(em,xyz);mk=crops._cached_block(seg,xyz,rid)
        if not mk.any():raise ValueError(f'Empty mask: root {rid}, node {node}')
        image[row,0]=im.transpose(2,1,0);mask[row]=np.packbits(mk.transpose(2,1,0).reshape(-1))
    for _ in readers.map(read,enumerate(ids)):pass
    # No chunk handles remain when another worker sees the durable batch.
    crops.close_batch_connections()
    temp=folder/f'.batch_{index:06d}.{os.getpid()}.tmp';temp.mkdir()
    np.save(temp/'image.npy',image,allow_pickle=False);np.save(temp/'mask.npy',mask,allow_pickle=False)
    np.save(temp/'node_ids.npy',ids,allow_pickle=False)
    atomic_json(temp/'metadata.json',dict(root_id=rid,materialization=MAT,coords_sha256=payload['coords_sha256'],
        n_nodes=len(ids),crop_size=129,batch_size=128))
    temp.replace(target);queue.complete(rid,index)
    print(f'INPUT READY root={rid} batch={index} nodes={len(ids)} seconds={time.monotonic()-started:.3f}',flush=True)


def collect(queue,rid,payload,held=None):
    if held is None:
        cell=queue.snapshot(rid)
        if cell['status']!='OPEN' or len(cell['done'])!=cell['n_batches']:return
    lock=held or queue.try_lock(f'{rid}_gc')
    if lock is None:return
    try:
        cell=queue.snapshot(rid)
        if cell['status']!='OPEN':return
        prefix=queue.prefix(cell)
        # Keep all cached chunks until the complete cell is durable.
        complete=prefix==cell['n_batches']
        if complete:
            folder=BASE/'inputs_v1'/str(rid)
            # All batches are durable, so all shared cache handles are closed.
            shutil.rmtree(BASE/'chunks_v1196_v1'/'cells'/str(rid),ignore_errors=True)
            shutil.rmtree(BASE/'chunks_v1196_v1'/'mask_mip2'/str(rid),ignore_errors=True)
            if (BASE/'chunks_v1196_v1'/'cells'/str(rid)).exists() or (BASE/'chunks_v1196_v1'/'mask_mip2'/str(rid)).exists():raise RuntimeError('Cell cache cleanup incomplete')
            atomic_json(folder/'ready.json',dict(root_id=rid,materialization=MAT,n_nodes=cell['n_nodes'],
                n_batches=cell['n_batches'],coords_sha256=payload['coords_sha256']))
            queue.budget.ready(rid,prepared_bytes(cell['n_nodes']))
            print(f'INPUT ROOT READY {rid} nodes={cell["n_nodes"]} batches={cell["n_batches"]}',flush=True)
        queue.gc_complete(rid,prefix,ready=complete)
    finally:lock.close()


def work(args):
    queue=BatchQueue(BASE);manifest=json.loads((BASE/'manifest.json').read_text());crops=PackedCrops(manifest)
    cache=SkeletonCache(st.open_store(DB_ROOT,'v1dd'));loaded={};deadline=time.monotonic()+args.max_seconds;built=0
    try:
        with ThreadPoolExecutor(max_workers=16) as readers:
            while time.monotonic()<deadline:
                claim=queue.claim()
                if claim is None:time.sleep(5);continue
                kind,rid,index,handle=claim
                if kind=='finished':return
                try:
                    if kind=='setup':
                        setup_plan(queue,crops,cache,rid);continue
                    if rid not in loaded:
                        # Bound RAM used by root plans in a long-lived worker.
                        loaded.clear();loaded[rid]=load_plan(queue,rid)
                    payload=loaded[rid]
                    if kind=='finalize':collect(queue,rid,payload,held=handle);handle=None;continue
                    print(f'BATCH CLAIM root={rid} batch={index} job={os.environ.get("SLURM_JOB_ID","local")}',flush=True)
                    prepare_batch(queue,crops,payload,rid,index,readers);built+=1
                finally:
                    if handle:handle.close()
                collect(queue,rid,payload)
                if args.limit_batches and built>=args.limit_batches:return
        return False
    finally:
        crops.chunk_pool.shutdown(wait=True);crops.close_batch_connections()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initialize',action='store_true');ap.add_argument('--max-seconds',type=float,default=42000)
    ap.add_argument('--limit-batches',type=int,default=0);args=ap.parse_args()
    if args.initialize:initialize()
    elif work(args) is False:raise SystemExit(75)
