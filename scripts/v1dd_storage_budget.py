"""Cross-process reservations for complete V1DD cells, protected by flock."""
import fcntl
from v1dd_file_lock import acquire_lock
import json
import os
from pathlib import Path

GIB=1024**3

def tree_bytes(path):
    total=0
    for directory,_,files in os.walk(path):
        for name in files:
            try: total+=os.stat(Path(directory)/name).st_blocks*512
            except FileNotFoundError: pass
    return total

class StorageBudget:
    def __init__(self,base):
        self.base=Path(base)
        self.path=self.base/'storage_budget.json'
        self.lock=self.base/'storage_budget.lock'

    def _change(self,callback):
        with self.lock.open('a') as handle:
            acquire_lock(handle)
            state=json.loads(self.path.read_text())
            before=json.dumps(state,separators=(',',':'))
            result=callback(state)
            if json.dumps(state,separators=(',',':'))==before:return result
            temp=self.path.with_suffix('.tmp.json')
            temp.write_text(json.dumps(state,indent=2)+'\n')
            temp.replace(self.path)
            return result

    def initialize(self,limit_bytes):
        owned={}
        for parent in [self.base/'inputs_v1',self.base/'chunks_v1196_v1'/'cells',self.base/'chunks_v1196_v1'/'mask_mip2']:
            if parent.exists():
                for folder in parent.iterdir():
                    if folder.is_dir(): owned[folder.name]=owned.get(folder.name,0)+tree_bytes(folder)
        total=tree_bytes(self.base)
        state=dict(limit_bytes=int(limit_bytes),common_bytes=max(0,total-sum(owned.values())),
                   cells={rid:dict(bytes=size,reserved=False) for rid,size in owned.items()})
        with self.lock.open('a') as handle:
            acquire_lock(handle)
            self.path.write_text(json.dumps(state,indent=2)+'\n')
        return state

    def reserve(self,rid,amount):
        def update(s):
            key=str(rid); old=s['cells'].get(key,dict(bytes=0,reserved=False))
            used=s['common_bytes']+sum(v['bytes'] for v in s['cells'].values())
            requested=int(amount) if old['reserved'] else max(int(amount),old['bytes'])
            if used-old['bytes']+requested>s['limit_bytes']: return False
            s['cells'][key]=dict(bytes=requested,reserved=True)
            return True
        return self._change(update)

    def ready(self,rid,amount):
        def update(s):
            # GPU deletion/commit can finish before CPU cleanup returns.
            if not (self.base/'inputs_v1'/str(rid)).exists(): amount_now=0
            else: amount_now=int(amount)
            s['cells'][str(rid)]=dict(bytes=amount_now,reserved=False)
        self._change(update)

    def release(self,rid):
        self._change(lambda s:s['cells'].update({str(rid):dict(bytes=0,reserved=False)}))


def prepared_bytes(n_nodes):
    n_batches=(n_nodes+127)//128
    return int(n_nodes*(129**3+(129**3+7)//8)+n_batches*32768)


def projected_cell_bytes(crops,centers):
    """Reserve full-cell inputs plus every unique image/mask storage chunk."""
    import numpy as np
    total=prepared_bytes(len(centers))
    for volume,packed in zip(crops._volumes(),[False,True]):
        size=np.asarray(volume.chunk_size,dtype=np.int64)
        offset=np.asarray(volume.voxel_offset,dtype=np.int64)
        start=np.maximum(centers-64,volume.bounds.minpt)
        end=np.minimum(centers+65,volume.bounds.maxpt)
        first=(start-offset)//size
        last=(end-1-offset)//size
        boxes=[]
        for x in range(int(np.max(last[:,0]-first[:,0]))+1):
            for y in range(int(np.max(last[:,1]-first[:,1]))+1):
                for z in range(int(np.max(last[:,2]-first[:,2]))+1):
                    shifted=first+[x,y,z]
                    valid=np.all(shifted<=last,axis=1)&np.all(start<end,axis=1)
                    boxes.append(shifted[valid])
        count=len(np.unique(np.concatenate(boxes),axis=0))
        per_chunk=int(np.prod(size))*(1 if packed else np.dtype(volume.dtype).itemsize)
        if packed: per_chunk=(per_chunk+7)//8
        # SQLite pages, B-tree, journals, and temporary write headroom.
        total+=int(count*(per_chunk+8192)*1.20)
    return total
