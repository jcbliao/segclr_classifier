"""Profile cold and warm V1DD preparation without modifying production caches."""
import sys, json, time, threading, tempfile, resource, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, '/orcd/home/002/jcbliao/rotation/segclr/segclr_db_named_v5/_pkgroot')
import infer_v1dd_candidates as pipeline
import numpy as np
from segclr_db import store as st
from segclr_db.skeletons import SkeletonCache
from cloudfiles import CloudFiles
from cloudvolume import chunks

ap=argparse.ArgumentParser()
ap.add_argument('--output',required=True)
ap.add_argument('--nodes',type=int,default=128)
args=ap.parse_args()
metrics=defaultdict(lambda:[0,0.,0.])
lock=threading.Lock()
def wrap(obj,name,label):
    original=getattr(obj,name)
    def measured(*a,**kw):
        start=time.perf_counter(); cpu=time.thread_time()
        try: return original(*a,**kw)
        finally:
            with lock:
                row=metrics[label]; row[0]+=1; row[1]+=time.perf_counter()-start; row[2]+=time.thread_time()-cpu
    setattr(obj,name,measured)
wrap(CloudFiles,'get','remote_fetch_including_transport_decompression')
wrap(chunks,'decode','chunk_decode')
wrap(np,'load','disk_numpy_read')
wrap(np,'save','disk_numpy_write')
wrap(np,'packbits','mask_pack')
wrap(np,'unpackbits','mask_unpack')
wrap(pipeline.fcntl,'flock','file_lock')
wrap(pipeline.V1DDCrops,'_mask','root_membership_mask')
wrap(pipeline.V1DDCrops,'_cached_block','crop_assembly_including_wait')
# download includes remote_fetch and chunk_decode; nested timings must not be summed.
from cloudvolume.datasource.precomputed.image import PrecomputedImageSource
wrap(PrecomputedImageSource,'download','volume_download_total')
manifest=json.loads((pipeline.BASE/'manifest.json').read_text())
crops=pipeline.V1DDCrops(manifest)
cache=SkeletonCache(st.open_store(pipeline.DB_ROOT,'v1dd'))
roots=pipeline.pd.read_parquet(pipeline.BASE/'cohort.parquet').pt_root_id.to_numpy()
results=[]
with tempfile.TemporaryDirectory(prefix='v1dd_profile_') as directory:
    crops.disk_cache=Path(directory)/'chunks'; (crops.disk_cache/'locks').mkdir(parents=True)
    for root in roots[:3]:
        rid=int(root); crops.prepare_root(rid)
        skeleton=cache.get_skeleton(rid,fetch_if_missing=False)
        centers=crops.nm_to_voxel(skeleton.coords)
        ids=pipeline.prefetch_order(centers)[:args.nodes]
        for phase in ['cold_remote','warm_disk','warm_ram']:
            if phase!='warm_ram':
                with crops.chunk_lock: crops.chunk_cache.clear(); crops.chunk_bytes=0
            metrics.clear(); started=time.perf_counter(); before=resource.getrusage(resource.RUSAGE_SELF)
            def read(node):
                em,seg=crops._volumes()
                image=crops._cached_block(em,centers[node])
                mask=crops._cached_block(seg,centers[node],rid)
                return image.transpose(2,1,0).copy(),np.packbits(mask.transpose(2,1,0).reshape(-1))
            with ThreadPoolExecutor(max_workers=16) as readers:
                data=list(readers.map(read,ids))
            assembled=time.perf_counter()
            images=np.stack([x[0] for x in data]); masks=np.stack([x[1] for x in data])
            np.save(Path(directory)/'image.npy',images); np.save(Path(directory)/'mask.npy',masks)
            elapsed=time.perf_counter()-started; after=resource.getrusage(resource.RUSAGE_SELF)
            row=dict(root_id=rid,phase=phase,nodes=len(ids),elapsed_s=elapsed,assembly_wall_s=assembled-started,
                inputs_per_min=60*len(ids)/elapsed,cpu_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                phases={k:dict(calls=v[0],summed_thread_wall_s=v[1],summed_thread_cpu_s=v[2]) for k,v in metrics.items()})
            results.append(row); print(json.dumps(row),flush=True)
            Path(args.output).write_text(json.dumps(dict(note='Nested concurrent timings overlap; do not sum. Warm disk may include OS page cache. Fresh node-local cache; production shared-filesystem I/O differs.',results=results),indent=2)+'\n')
            del data,images,masks
crops.chunk_pool.shutdown()
