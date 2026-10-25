"""Pack whole-cell windows without changing their original extraction or sampling."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from data.packed_presynaptic import PackedPresynapticWindowDataset


_PACK_CONTEXT = None

def _extract_packing_chunk(start):
    source, fields, targets, centers, ptr, names = _PACK_CONTEXT
    stop = min(start + 256, len(source))
    windows = [source[i] for i in range(start, stop)]
    batch = Batch.from_data_list(windows)
    lo, hi = ptr[start], ptr[stop]
    for name in names:
        fields[name][lo:hi] = batch[name].numpy()
    targets[start:stop] = batch.y_levels.numpy()
    centers[start:stop] = [int(w.center_local_idx) for w in windows]
    counts = np.asarray([w.edge_index.shape[1] for w in windows])
    local_edges = batch.edge_index.numpy() - np.repeat(batch.ptr[:-1].numpy(), counts)[None, :]
    attr = getattr(batch, 'edge_attr', None)
    return local_edges, None if attr is None else attr.numpy(), counts


class PackedWholeCellWindows(PackedPresynapticWindowDataset):
    def __init__(self, source):
        self.source = source
        self.index_root_ids = source.index_root_ids
        self.index_labels = source.index_labels
        self.hierarchy = source.hierarchy
        self.classes = source.classes
        self.window_sizes = np.empty(len(source), dtype=np.int64)
        roots, starts, counts = np.unique(self.index_root_ids,return_index=True,return_counts=True)
        for root,start,count in zip(roots,starts,counts):
            rows = slice(start,start+count)
            offsets, _ = source.cell_membership[int(root)]
            centers = (source.index_membership_centers[rows] if source.num_embeddings is not None
                       else source.index_centers[rows])
            self.window_sizes[rows] = offsets[centers + 1] - offsets[centers]

    def __len__(self):
        return len(self.source)

    def __getattr__(self, name):
        # Export/analysis code still sees the original center and cell metadata.
        if name == 'source':
            raise AttributeError(name)
        return getattr(self.source, name)

    def __getitem__(self, index):
        return self.source[index]

    def get_batch(self, indices, device=False):
        if not hasattr(self, '_packed'):
            return Batch.from_data_list([self.source[int(i)] for i in indices])
        batch = super().get_batch(indices, device)
        batch.center_local_idx = torch.from_numpy(self._packed['centers'][indices].copy())
        batch.window_size = torch.from_numpy(self.window_sizes[indices].copy())
        if self._packed['edge_attr'] is not None:
            from data.batched_presynaptic import ragged_indices
            offsets = self._packed['edge_offsets']
            ids = ragged_indices(offsets[indices], offsets[np.asarray(indices) + 1] - offsets[indices])
            batch.edge_attr = torch.from_numpy(self._packed['edge_attr'][ids])
        return batch

    def prepare_packed(self, workers=8, max_bytes=44 * 2**30):
        started = time.perf_counter()
        ptr = np.r_[0, np.cumsum(self.window_sizes)]
        sample = self.source[0]
        names = [name for name in ('x', 'pos', 'rel_pos', 'pos_enc', 'thickness')
                 if getattr(sample, name, None) is not None]
        estimate = int(ptr[-1]) * (sum(sample[name][0].numel() * sample[name].element_size()
                                      for name in names) + 128) + len(self) * 128
        if estimate > max_bytes:
            raise MemoryError(f'Packed whole-cell windows need approximately {estimate/2**30:.1f} GiB')
        workdir = getattr(self, '_packing_workdir', None)
        def allocate(name, shape, dtype):
            if workdir is None: return np.empty(shape, dtype=dtype)
            path=Path(workdir)/f'{name}.npy';path.parent.mkdir(exist_ok=True)
            return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=tuple(map(int,shape)))
        fields = {name: allocate(f'fields/{name}', (ptr[-1], *sample[name].shape[1:]), sample[name].numpy().dtype)
                  for name in names}
        fields['has_segclr'] = allocate('fields/has_segclr', (ptr[-1],), bool)
        fields['has_segclr'][:] = True
        targets = allocate('targets', (len(self), self.hierarchy.depth), np.int64)
        centers = allocate('centers', (len(self),), np.int64)
        edges, attributes, edge_counts = [], [], []
        global _PACK_CONTEXT
        _PACK_CONTEXT = (self.source, fields, targets, centers, ptr, names)
        chunks = range(0, len(self), 256)
        # Processes write disjoint regions of shared disk-backed arrays. Fork happens
        # only in CPU packing jobs, before CUDA initialization.
        pool_class = ProcessPoolExecutor if workdir is not None and not torch.cuda.is_initialized() else ThreadPoolExecutor
        kwargs = dict(max_workers=workers)
        if pool_class is ProcessPoolExecutor: kwargs['mp_context']=multiprocessing.get_context('fork')
        try:
            with pool_class(**kwargs) as pool:
                results=pool.map(_extract_packing_chunk, chunks, chunksize=8)
                for edge, attr, counts in tqdm(results,total=len(chunks),desc='Packing whole-cell chunks',mininterval=10):
                    edges.append(edge); attributes.append(attr); edge_counts.append(counts)
        finally:
            _PACK_CONTEXT = None
        self._packed = dict(fields=fields, ptr=ptr, targets=targets, centers=centers,
                            synapses=None, post=None, edges=np.concatenate(edges, axis=1),
                            edge_offsets=np.r_[0, np.cumsum(np.concatenate(edge_counts))],
                            edge_attr=None if attributes[0] is None else np.concatenate(attributes))
        self.packing_bytes = sum(v.nbytes for v in fields.values()) + sum(
            v.nbytes for v in self._packed.values() if isinstance(v, np.ndarray))
        self.packing_seconds = time.perf_counter() - started
        print(f'Packed {len(self):,} whole-cell windows in {self.packing_seconds:.1f}s '
              f'({self.packing_bytes/2**30:.2f} GiB)', flush=True)
        return True
