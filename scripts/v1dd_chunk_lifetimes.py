"""Chunk lifetimes and peak storage for Morton-ordered crop batches."""
import numpy as np
from v1dd_storage_budget import prepared_bytes

class ChunkLifetimes:
    def __init__(self,crops,centers,order,batch_size=128,window=1):
        self.n_nodes=len(order)
        self.batch_size=batch_size
        self.window=window
        self.n_batches=(len(order)+batch_size-1)//batch_size
        self.items={}
        self.expired={}
        self.batch_keys={}
        self.first={}
        self.full_cache_bytes=0
        self.last={}
        events=np.zeros(self.n_batches+1,dtype=np.int64)
        centers=centers[order]
        rows=np.arange(len(centers))//batch_size
        for volume,packed in zip(crops._volumes(),[False,True]):
            size=np.asarray(volume.chunk_size,dtype=np.int64)
            offset=np.asarray(volume.voxel_offset,dtype=np.int64)
            start=np.maximum(centers-64,volume.bounds.minpt)
            end=np.minimum(centers+65,volume.bounds.maxpt)
            first=(start-offset)//size
            last=(end-1-offset)//size
            boxes=[];batches=[]
            if not len(centers):continue
            for x in range(int(np.max(last[:,0]-first[:,0]))+1):
                for y in range(int(np.max(last[:,1]-first[:,1]))+1):
                    for z in range(int(np.max(last[:,2]-first[:,2]))+1):
                        shifted=first+[x,y,z]
                        valid=np.all(shifted<=last,axis=1)&np.all(start<end,axis=1)
                        boxes.append(shifted[valid]);batches.append(rows[valid])
            unique,inverse=np.unique(np.concatenate(boxes),axis=0,return_inverse=True)
            used=np.concatenate(batches)
            first_use=np.full(len(unique),self.n_batches,dtype=np.int64)
            last_use=np.full(len(unique),-1,dtype=np.int64)
            np.minimum.at(first_use,inverse,used)
            np.maximum.at(last_use,inverse,used)
            names=[]
            for grid,first_batch,last_batch in zip(unique,first_use,last_use):
                lo=np.maximum(grid*size+offset,volume.bounds.minpt)
                hi=np.minimum(grid*size+offset+size,volume.bounds.maxpt)
                name='_'.join(map(str,tuple(lo)+tuple(hi)))
                key=('mask:' if packed else 'image:')+name
                raw=int(np.prod(hi-lo))
                payload=(raw+7)//8 if packed else raw*np.dtype(volume.dtype).itemsize
                # SQLite pages/journal headroom and per-chunk index overhead.
                charged=int((payload+8192)*1.20)
                names.append(key)
                self.full_cache_bytes+=charged
                self.items[key]=(packed,tuple(lo),tuple(hi))
                self.first[key]=int(first_batch);self.last[key]=int(last_batch)
                self.expired.setdefault(int(last_batch),[]).append(key)
                events[max(0,first_batch-window+1)]+=charged;events[last_batch+1]-=charged
            for batch,index in np.unique(np.column_stack((used,inverse)),axis=0):
                self.batch_keys.setdefault(int(batch),[]).append(names[index])
        live=np.cumsum(events)[:-1]
        inputs=np.array([prepared_bytes(min((i+window)*batch_size,len(order))) for i in range(self.n_batches)],dtype=np.int64)
        self.peak_chunk_bytes=int(live.max(initial=0))
        self.peak_total_bytes=int((live+inputs).max(initial=0))
