"""Pack the actual selected windows once, then gather batches without root decoding."""
import copy
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from data.batched_presynaptic import BatchedPresynapticWindowDataset, ragged_indices
from data import packed_segmentation_augmentation as segmentation
from data.segmentation_augmentation import training_choice_indices

class PackedPresynapticWindowDataset(BatchedPresynapticWindowDataset):
    def prepare_packed(self, workers=8, max_bytes=44 * 2**30):
        if self.embedding_augmentation is not None and not segmentation.enabled(self):
            return False  # stochastic policies retain their original RNG path
        started=time.perf_counter()
        geometry=getattr(self,'geometry_required',True)
        ptr=np.r_[0,np.cumsum(self.window_sizes,dtype=np.int64)]
        # Conservative allowance includes intermediate copies and geometry/edges.
        estimate=int(ptr[-1])*(384 if geometry else 272)+len(self)*128
        if estimate>max_bytes:
            raise MemoryError(f'Packed windows need approximately {estimate/2**30:.1f} GiB')
        clean=copy.copy(self);clean.embedding_augmentation=None;clean._cache=OrderedDict()
        sample=BatchedPresynapticWindowDataset.__getitems__(clean,[0])
        fields={'x':np.empty((ptr[-1],sample.x.shape[1]),np.float32),'has_segclr':np.empty(ptr[-1],bool)}
        if geometry:
            fields.update(pos=np.empty((ptr[-1],3),np.float32),rel_pos=np.empty((ptr[-1],3),np.float32),pos_enc=np.empty((ptr[-1],sample.pos_enc.shape[1]),np.float32))
        targets=np.empty((len(self),self.hierarchy.depth),np.int64)
        centers=np.empty(len(self),np.int64)
        synapses=np.empty(len(self),np.int64) if self.postsynaptic_cache is not None else None
        post=np.empty((len(self),64),np.float32) if self.use_postsynaptic else None
        roots,starts,counts=np.unique(self.index_root_ids,return_index=True,return_counts=True)
        tasks=sorted(zip(roots,starts,counts),key=lambda t:t[1])
        def pack(task):
            root,start,count=map(int,task);end=start+count
            if not np.all(self.index_root_ids[start:end]==root):raise ValueError('Root rows must be contiguous')
            local=copy.copy(self);local._cache=OrderedDict();local.cell_cache_size=1;local.embedding_augmentation=None
            batch=BatchedPresynapticWindowDataset.__getitems__(local,np.arange(start,end))
            lo,hi=int(ptr[start]),int(ptr[end])
            for name,array in fields.items():array[lo:hi]=batch[name].numpy()
            targets[start:end]=batch.y_levels.numpy()
            centers[start:end]=batch.center_index.numpy()-batch.ptr[:-1].numpy()
            if synapses is not None:synapses[start:end]=batch.synapse_id.numpy()
            if post is not None:post[start:end]=batch.postsynaptic_embedding.numpy()
            return batch.edge_index.numpy()+lo
        with ThreadPoolExecutor(max_workers=workers) as pool:
            edges=list(tqdm(pool.map(pack,tasks),total=len(tasks),desc="Packing presynaptic cells",mininterval=10))
        edge_index=np.concatenate(edges,axis=1) if edges else np.empty((2,0),np.int64)
        edge_graph=np.searchsorted(ptr[1:],edge_index[0],side='right')
        edge_offsets=np.r_[0,np.cumsum(np.bincount(edge_graph,minlength=len(self)))]
        if edge_index.shape[1]:edge_index=edge_index-ptr[edge_graph][None,:]
        self._packed=dict(fields=fields,ptr=ptr,targets=targets,centers=centers,synapses=synapses,post=post,edges=edge_index,edge_offsets=edge_offsets)
        self._cache.clear()
        self.packing_seconds=time.perf_counter()-started
        self.packing_bytes=sum(v.nbytes for v in fields.values())+edge_index.nbytes+targets.nbytes+centers.nbytes+ptr.nbytes+edge_offsets.nbytes
        if segmentation.enabled(self):
            segmentation.attach(self,workers,max_bytes)
        print(f'Packed {len(self):,} windows in {self.packing_seconds:.1f}s ({self.packing_bytes/2**30:.2f} GiB)',flush=True)
        return True

    def prepare_device(self,device,reserve_bytes=8*2**30):
        required=sum(v.nbytes for v in self._packed["fields"].values())
        if 'augmentation_bank' in self._packed:
            required+=self._packed['augmentation_bank'].nbytes+self._packed['augmentation_rows'].nbytes
        free,_=torch.cuda.mem_get_info(device)
        if required+reserve_bytes>free:
            return False
        self._device=device
        fields={name:torch.from_numpy(array).to(device) for name,array in self._packed["fields"].items()}
        if 'augmentation_bank' in self._packed:
            bank=torch.from_numpy(self._packed['augmentation_bank']).to(device)
            rows=torch.from_numpy(self._packed['augmentation_rows']).to(device)
            columns=torch.tensor(training_choice_indices(self.embedding_augmentation,self._packed['augmentation_labels'].tolist()),dtype=torch.long,device=device)
            self._packed.update(device_augmentation_bank=bank,device_augmentation_rows=rows,device_augmentation_columns=columns)
        self._packed["device_fields"]=fields
        print(f"Resident window features: {required/2**30:.2f} GiB on {device}",flush=True)
        return True

    def __getitems__(self,indices):
        return self.get_batch(indices,device=False)

    def get_batch(self,indices,device=False):
        if not hasattr(self,'_packed'):return super().__getitems__(indices)
        indices=np.asarray(indices,np.int64)
        if not len(indices):raise ValueError('Empty batch')
        p=self._packed;sizes=self.window_sizes[indices].astype(np.int64);ptr=np.r_[0,np.cumsum(sizes)]
        nodes=ragged_indices(p['ptr'][indices],sizes)
        if device:
            node_ids=torch.from_numpy(nodes).pin_memory().to(self._device,non_blocking=True)
            values={name:array.index_select(0,node_ids) for name,array in p['device_fields'].items()}
        else:
            values={name:torch.from_numpy(array[nodes]) for name,array in p['fields'].items()}
        if 'augmentation_bank' in p:
            values['x']=self.sample_packed_x(node_ids if device else nodes,device=device)
        lengths=p['edge_offsets'][indices+1]-p['edge_offsets'][indices]
        edge_ids=ragged_indices(p['edge_offsets'][indices],lengths)
        edges=p['edges'][:,edge_ids]+np.repeat(ptr[:-1],lengths)[None,:]
        result=Batch(**values,edge_index=torch.from_numpy(edges),y_levels=torch.from_numpy(p['targets'][indices]),
                     root_id=torch.from_numpy(self.index_root_ids[indices]),center_index=torch.from_numpy(p['centers'][indices]+ptr[:-1]),
                     batch=torch.from_numpy(np.repeat(np.arange(len(indices)),sizes)),ptr=torch.from_numpy(ptr))
        result.y=result.y_levels[:,-1]
        if p['synapses'] is not None:result.synapse_id=torch.from_numpy(p['synapses'][indices])
        if p['post'] is not None:result.postsynaptic_embedding=torch.from_numpy(p['post'][indices])
        return result

    def sample_packed_x(self,indices,device=False,choices=None):
        return segmentation.sample(self,indices,device=device,choices=choices)

class ResidentWindowLoader:
    """The original sampler with GPU gathers; CPU metadata remains available for routing."""
    def __init__(self,dataset,batch_sampler,generator=None):
        self.dataset=dataset;self.batch_sampler=batch_sampler;self.generator=generator
    def __len__(self):return len(self.batch_sampler)
    def __iter__(self):
        if self.generator is not None:
            torch.empty((),dtype=torch.int64).random_(generator=self.generator)
        for indices in self.batch_sampler:
            yield self.dataset.get_batch(indices,device=True)
