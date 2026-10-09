"""CPU post-site crops using shared packed chunks and retained ownership."""
import json,zlib,time
import requests
from pathlib import Path
import numpy as np
from v1dd_postsynaptic_crops import V1DDPostCrops
from v1dd_packed_chunks import PackedCrops
from v1dd_file_lock import acquire_lock
from v1dd_api_gate import retry_api

CACHE_CAP=16*1024**3
class SharedPostCrops(V1DDPostCrops,PackedCrops):
    def __init__(self,manifest):
        # Imports touch many files on home NFS. Do not occupy an API slot while
        # loading Python packages; CPU preparation deliberately avoids torch.
        import cloudvolume
        # Reader volumes reuse metadata from initialization.
        def initialize():
            if hasattr(self,'chunk_pool'):self.chunk_pool.shutdown(wait=True)
            super(SharedPostCrops,self).__init__(manifest)
        retry_api(initialize,'volume_metadata')
        session=self.client.chunkedgraph.session
        from requests.adapters import HTTPAdapter
        session.mount('https://',HTTPAdapter(max_retries=0))
        request=session.request
        def bounded_request(method,url,**kwargs):
            kwargs.setdefault('timeout',(10,30))
            return request(method,url,**kwargs)
        session.request=bounded_request

    def select_shard(self,base,part):
        if getattr(self,'active_root',None)==part:return
        if getattr(self,'raw_pending',{}):raise RuntimeError('Cannot change shard with active readers')
        self.close_batch_connections()
        self.disk_cache=Path(base)/'chunks_v1196_v1';(self.disk_cache/'locks').mkdir(parents=True,exist_ok=True)
        self.active_root=part
        with self.raw_guard:self.raw.clear();self.raw_bytes=0

    def _download_chunk(self,volume,start,end,segmentation):
        shape=tuple(map(int,end-start));dtype=np.dtype(np.uint64 if segmentation else volume.dtype)
        key=('supervoxels:' if segmentation else 'image:')+'_'.join(map(str,tuple(start)+tuple(end)))
        pack=self._packed_cache()
        def decode(payload):
            data=np.frombuffer(zlib.decompress(payload),dtype=dtype)
            if data.size!=int(np.prod(shape)):raise ValueError('Packed chunk shape mismatch')
            return data.reshape(shape)
        payload=pack.get(key)
        if payload is not None:return decode(payload)
        with self._chunk_ownership(key):
            payload=pack.get(key)
            if payload is not None:return decode(payload)
            data=super()._download_chunk(volume,start,end,segmentation)
            payload=zlib.compress(data.tobytes(),level=1)
            self._store_payload(pack,key,payload)
            return data

    def _store_payload(self,pack,key,payload):
        # Compression bounds actual disk writes, while the scheduler reserves
        # the full cap plus complete prepared inputs before admitting a shard.
        folder=pack.folder.parent;counter=folder/'cache_bytes.json'
        with (folder/'cache_bytes.lock').open('a') as lock:
            acquire_lock(lock);used=json.loads(counter.read_text()) if counter.exists() else 0
            permitted=used+len(payload)+256<=CACHE_CAP
            if permitted:
                temp=counter.with_suffix('.tmp.json');temp.write_text(json.dumps(used+len(payload)+256));temp.replace(counter)
        if permitted:self._write_cache(None,key,payload)
        else:
            # Correct, bounded fallback: keep only the worker's RAM copy.
            # Never wait for space that can be freed only after shard completion.
            print(f'POST CACHE CAP shard={self.active_root}: using RAM for uncached chunks',flush=True)

    def _resolve_labels(self,ids):
        missing=[int(i) for i in ids if int(i) not in self.root_map]
        def request(subset):
            self.root_requests=getattr(self,'root_requests',0)+1
            started=time.monotonic()
            try:
                roots=retry_api(lambda:self.client.chunkedgraph.get_roots(subset,timestamp=self.timestamp),
                                f'target_validation_{len(subset)}',on_wait=self._packed_cache().flush)
            except requests.HTTPError as exc:
                # Only explicit payload-size failures justify more requests.
                # 429/5xx/timeouts back off on the same request; splitting those
                # failures amplifies load precisely when the server is struggling.
                if exc.response is None or exc.response.status_code!=413 or len(subset)<=128:raise
                middle=len(subset)//2
                print(f'POST ROOT SPLIT labels={len(subset)} error=HTTP_413',flush=True)
                return np.concatenate([request(subset[:middle]),request(subset[middle:])])
            if len(roots)!=len(subset):raise ValueError('Root API returned a different number of labels')
            print(f'POST ROOT REQUEST labels={len(subset)} seconds={time.monotonic()-started:.3f}',flush=True)
            return roots
        for start in range(0,len(missing),4096):
            subset=missing[start:start+4096];roots=request(subset)
            self.root_map.update(zip(subset,map(int,roots)))
        return np.asarray([self.root_map[int(i)] for i in ids],dtype=np.uint64)

    def _bounded_candidates(self,center,root_id,sv):
        """Only target-root leaves intersecting the physical crop, as uint64."""
        if not hasattr(self,'invalid_target_roots'):self.invalid_target_roots=set()
        if int(root_id) in self.invalid_target_roots:return np.empty(0,np.uint64)
        _,seg=self._volumes()
        lo=np.maximum(np.asarray(center,dtype=np.int64)-64,seg.bounds.minpt)
        hi=np.minimum(np.asarray(center,dtype=np.int64)+65,seg.bounds.maxpt)
        if np.any(lo>=hi):return np.empty(0,np.uint64)
        # CAVE bounds use chunkedgraph base-resolution coordinates, not mip voxels.
        base=np.asarray(self.client.chunkedgraph.base_resolution,dtype=float)
        bounds=np.stack([np.floor(lo*self.resolution/base),np.ceil(hi*self.resolution/base)],axis=1).astype(np.int64)
        key=f'target1196:{int(root_id)}:'+','.join(map(str,bounds.reshape(-1)))
        pack=self._packed_cache();payload=pack.get(key)
        if payload is None:
            with self._chunk_ownership(key):
                payload=pack.get(key)
                if payload is None:
                    self.leaf_requests=getattr(self,'leaf_requests',0)+1
                    try:
                        leaves=retry_api(lambda:self.client.chunkedgraph.get_leaves(int(root_id),bounds=bounds,stop_layer=1),
                                         'bounded_target_leaves',on_wait=pack.flush)
                    except requests.HTTPError as exc:
                        response=exc.response
                        if response is None or response.status_code!=401 or 'root_id not valid at timestamp' not in response.text:raise
                        # The public table rejects IDs newer than its snapshot.
                        # Such roots cannot match timestamped public segmentation.
                        self.invalid_target_roots.add(int(root_id));leaves=np.empty(0,np.uint64)
                        print(f'POST INVALID TARGET ROOT {int(root_id)}: empty historical mask',flush=True)
                    leaves=np.unique(np.asarray(leaves,dtype=np.uint64));leaves=leaves[leaves!=0]
                    payload=zlib.compress(leaves.tobytes(),level=1);self._store_payload(pack,key,payload)
        leaves=np.frombuffer(zlib.decompress(payload),dtype=np.uint64)
        # Match uint64 on both sides: signed/unsigned mixing can lose ID precision.
        return np.intersect1d(np.unique(sv),leaves,assume_unique=True)

    def prepare_targets(self,root_ids):
        if len(self.root_map)>1000000:self.root_map.clear()

    def fetch_crop(self,center,root_id):
        em,seg=self._volumes();center=np.asarray(center,dtype=np.int64)
        image=self._site_block(em,center);sv=self._site_block(seg,center,True)
        ids=self._bounded_candidates(center,root_id,sv)
        with self.root_guard:roots=self._resolve_labels(ids)
        mask=np.isin(sv,ids[roots==int(root_id)])
        return image.transpose(2,1,0)[None],mask.transpose(2,1,0)[None]

    def fill_batch(self,centers,roots,image,masks,statuses,readers):
        """Download in parallel; map crop labels once per 32-site group.

        Only bounded target leaves receive historical root validation.
        Root RPCs are serial within a worker. Raw segmentation memory is bounded
        to 32 crops (~546 MiB) rather than all 128 crops (~2.1 GiB).
        """
        import time
        self.prepare_targets(roots);download_seconds=lookup_seconds=mask_seconds=0.;rpc_before=len(self.root_map);requests_before=getattr(self,'root_requests',0);leaves_before=getattr(self,'leaf_requests',0)
        for start in range(0,len(roots),32):
            indices=list(range(start,min(start+32,len(roots))));started=time.monotonic()
            def read(row):
                if roots[row]==0:statuses[row]='unresolved_root';return row,None
                em,seg=self._volumes();center=np.asarray(centers[row],dtype=np.int64)
                raw_image=self._site_block(em,center)
                image[row]=raw_image.transpose(2,1,0)[None]
                return row,self._site_block(seg,center,True)
            crops=list(readers.map(read,indices));download_seconds+=time.monotonic()-started
            # Publish completed chunks before any potentially slow root RPC.
            self._packed_cache().flush()
            started=time.monotonic();candidates={row:self._bounded_candidates(centers[row],roots[row],sv) for row,sv in crops if sv is not None}
            label_sets=list(candidates.values())
            ids=np.unique(np.concatenate(label_sets)) if label_sets else np.empty(0,np.uint64)
            ids=ids[ids!=0];mapped=self._resolve_labels(ids);lookup_seconds+=time.monotonic()-started
            started=time.monotonic()
            for row,sv in crops:
                if sv is None:continue
                chosen=candidates[row];positions=np.searchsorted(ids,chosen)
                wanted=chosen[mapped[positions]==int(roots[row])]
                mask=np.isin(sv,wanted)
                if not mask.any():statuses[row]='empty_mask';image[row]=0;continue
                masks[row]=np.packbits(mask.transpose(2,1,0).reshape(-1))
            mask_seconds+=time.monotonic()-started
            # Do not retain the previous group's crops while downloading another.
            del crops,label_sets,candidates
            sv=None
        print(f'POST BATCH PROFILE download_s={download_seconds:.3f} root_lookup_s={lookup_seconds:.3f} mask_s={mask_seconds:.3f} new_labels={len(self.root_map)-rpc_before} root_requests={getattr(self,"root_requests",0)-requests_before} leaf_requests={getattr(self,"leaf_requests",0)-leaves_before}',flush=True)
