"""Exact V1DD post-site crops with shared bounded raw-chunk and root caches."""
import threading
from concurrent.futures import Future
from collections import OrderedDict
import numpy as np
from infer_v1dd_candidates import V1DDCrops

class V1DDPostCrops(V1DDCrops):
    def __init__(self,manifest):
        super().__init__(manifest)
        self.raw=OrderedDict();self.raw_pending={};self.raw_bytes=0
        self.raw_guard=threading.Lock();self.root_guard=threading.Lock();self.root_map={}
        # This loader fills directly from crop threads, without a second pool.
        self.chunk_pool.shutdown(wait=True)

    def _download_chunk(self,volume,start,end,segmentation):
        from cloudvolume.lib import Bbox
        from cloudvolume.exceptions import EmptyVolumeException
        try:
            data=volume.download(Bbox(start,end),agglomerate=False) if segmentation else volume.download(Bbox(start,end))
        except EmptyVolumeException:
            if not segmentation:raise
            data=np.zeros(tuple(end-start),volume.dtype)
        data=np.asarray(data)
        return data[...,0] if data.ndim==4 else data

    def _raw_chunk(self,volume,start,end,segmentation):
        from cloudvolume.lib import Bbox
        from cloudvolume.exceptions import EmptyVolumeException
        key=(segmentation,tuple(start),tuple(end))
        with self.raw_guard:
            if key in self.raw:
                self.raw.move_to_end(key);return self.raw[key]
            future=self.raw_pending.get(key);owner=future is None
            if owner:self.raw_pending[key]=future=Future()
        if not owner:return future.result()
        try:
            data=self._download_chunk(volume,start,end,segmentation)
            data.setflags(write=False)
            with self.raw_guard:
                self.raw[key]=data;self.raw_bytes+=data.nbytes
                while self.raw_bytes>512*1024**2 and len(self.raw)>1:
                    _,old=self.raw.popitem(last=False);self.raw_bytes-=old.nbytes
                self.raw_pending.pop(key);future.set_result(data)
            return data
        except BaseException as exc:
            with self.raw_guard:self.raw_pending.pop(key);future.set_exception(exc)
            raise

    def _site_block(self,volume,center,segmentation=False):
        lo=np.asarray(center,dtype=np.int64)-64;hi=lo+129
        out=np.zeros((129,)*3,dtype=volume.dtype)
        size=np.asarray(volume.chunk_size,dtype=np.int64);offset=np.asarray(volume.voxel_offset,dtype=np.int64)
        first=(lo-offset)//size;last=(hi-1-offset)//size
        for x in range(first[0],last[0]+1):
            for y in range(first[1],last[1]+1):
                for z in range(first[2],last[2]+1):
                    start=np.maximum(offset+np.array([x,y,z])*size,volume.bounds.minpt)
                    end=np.minimum(offset+(np.array([x,y,z])+1)*size,volume.bounds.maxpt)
                    a=np.maximum(start,lo);b=np.minimum(end,hi)
                    if np.any(a>=b):continue
                    data=self._raw_chunk(volume,start,end,segmentation)
                    out[tuple(slice(int(u),int(v)) for u,v in zip(a-lo,b-lo))]=data[tuple(slice(int(u),int(v)) for u,v in zip(a-start,b-start))]
        return out

    def fetch_crop(self,center,root_id):
        import torch
        # SegCLRInference._embed converts nanometres to voxels before _fill.
        em,seg=self._volumes();center=np.asarray(center,dtype=np.int64)
        image=self._site_block(em,center);sv=self._site_block(seg,center,True)
        ids=np.unique(sv);ids=ids[ids!=0]
        with self.root_guard:
            missing=[int(i) for i in ids if int(i) not in self.root_map]
            for start in range(0,len(missing),4096):
                subset=missing[start:start+4096]
                roots=self.client.chunkedgraph.get_roots(subset,timestamp=self.timestamp)
                self.root_map.update(zip(subset,map(int,roots)))
            wanted=np.array([i for i in ids if self.root_map[int(i)]==int(root_id)],dtype=sv.dtype)
            if len(self.root_map)>1000000:self.root_map.clear()
        mask=np.isin(sv,wanted)
        return torch.from_numpy(image.transpose(2,1,0).copy())[None],torch.from_numpy(mask.transpose(2,1,0).copy())[None]
