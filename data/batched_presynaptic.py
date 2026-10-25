"""Bulk presynaptic reads with unchanged windows, features and batch order."""
import numpy as np
import torch
from torch_geometric.data import Batch
from data.dataset_presynaptic import PresynapticWindowDataset
from data.geodesic_window import REL_POS_SCALE_NM


def collate_presynaptic(batch):
    """torch DataLoader receives an already assembled Batch from __getitems__."""
    return batch if isinstance(batch, Batch) else Batch.from_data_list(batch)


def ragged_indices(starts, lengths):
    offsets=np.r_[0,np.cumsum(lengths)]
    return np.repeat(starts,lengths)+np.arange(offsets[-1])-np.repeat(offsets[:-1],lengths)


class BatchedPresynapticWindowDataset(PresynapticWindowDataset):
    def __init__(self, *args, geometry_required=True, **kwargs):
        self.geometry_required = geometry_required
        super().__init__(*args, **kwargs)

    def __getitems__(self, indices):
        # Preserve augmentation RNG call order exactly for existing stochastic policies.
        if self.embedding_augmentation is not None:
            return Batch.from_data_list([self[int(i)] for i in indices])
        indices=np.asarray(indices,dtype=np.int64)
        geometry_required=getattr(self,'geometry_required',True)
        if not len(indices):raise ValueError('Empty batch')
        roots=self.index_root_ids[indices];rows=self.index_window_rows[indices]
        sizes=self.window_sizes[indices].astype(np.int64);ptr=np.r_[0,np.cumsum(sizes)]
        total=int(ptr[-1]);first=self._load_cell(int(roots[0]))
        x=np.zeros((total,first['observed_x'].shape[1]),np.float32)
        has=np.zeros(total,bool)
        pos=np.empty((total,3),np.float32) if geometry_required else None
        rel=np.empty_like(pos) if geometry_required else None
        lpe=np.zeros((total,first['lpe'].shape[1]),np.float32) if geometry_required else None
        targets=np.empty((len(indices),first['targets'].shape[1]),np.int64)
        centers=np.empty(len(indices),np.int64);edges=[]
        synapses=np.empty(len(indices),np.int64) if self.postsynaptic_cache is not None else None
        post=np.empty((len(indices),64),np.float32) if self.use_postsynaptic else None
        # First-occurrence root order avoids changing cache access unnecessarily.
        for rid in dict.fromkeys(roots.tolist()):
            group=np.flatnonzero(roots==rid);cell=self._load_cell(int(rid));r=rows[group];lengths=sizes[group]
            if self.single_presynaptic_embedding:
                nodes=np.asarray(cell['centers'][r],dtype=np.int64);member_ids=None
            else:
                member_ids=ragged_indices(cell['offsets'][r],lengths)
                nodes=cell['members'][member_ids]
            destination=ragged_indices(ptr[group],lengths)
            if 'batched_observed_rows' not in cell:
                lookup=np.full(len(cell['pos']),-1,np.int64)
                lookup[cell['observed_ids']]=np.arange(len(cell['observed_ids']))
                cell['batched_observed_rows']=lookup
            observed=cell['batched_observed_rows'][nodes];valid=observed>=0
            if self.single_presynaptic_embedding and not valid.all():raise ValueError(f'Nearest observed node lacks an embedding: {rid}')
            x[destination[valid]]=cell['observed_x'][observed[valid]]
            has[destination]=valid
            center_nodes=cell['centers'][r]
            if geometry_required:
                positions=cell['pos'][nodes]
                pos[destination]=positions
                rel[destination]=(positions-cell['pos'][np.repeat(center_nodes,lengths)])/REL_POS_SCALE_NM
                if member_ids is not None:lpe[destination]=cell['lpe'][member_ids]
            targets[group]=cell['targets'].numpy();centers[group]=center_nodes
            if synapses is not None:synapses[group]=cell['synapse_ids'][r]
            if post is not None:post[group]=cell['post_x'][r]
            if self.single_presynaptic_embedding or not geometry_required:continue
            if 'window_edges' in cell:
                edge_lengths=cell['window_edge_offsets'][r+1]-cell['window_edge_offsets'][r]
                edge_ids=ragged_indices(cell['window_edge_offsets'][r],edge_lengths)
                edges.append(np.asarray(cell['window_edges'][edge_ids],dtype=np.int64)+np.repeat(ptr[group],edge_lengths)[:,None])
            else:
                offsets=np.r_[0,np.cumsum(lengths)]
                pairs=[]
                for g,lo,hi in zip(group,offsets[:-1],offsets[1:]):
                    window=nodes[lo:hi];local={int(node):i for i,node in enumerate(window)}
                    for node in window:
                        for neighbor in cell['adjacency'][int(node)]:
                            if neighbor in local:pairs.append((int(ptr[g])+local[int(node)],int(ptr[g])+local[neighbor]))
                if pairs:edges.append(np.asarray(pairs,np.int64))
        edge_array=np.concatenate(edges,axis=0).T.copy() if edges else np.empty((2,0),np.int64)
        # Grouped reads can reorder edges across graphs; restore original graph order.
        if edge_array.shape[1]:
            graph_of_edge=np.searchsorted(ptr[1:],edge_array[0],side='right')
            edge_array=edge_array[:,np.argsort(graph_of_edge,kind='stable')].copy()
        result=Batch(x=torch.from_numpy(x),edge_index=torch.from_numpy(edge_array),
                     has_segclr=torch.from_numpy(has),y_levels=torch.from_numpy(targets),
                     # Match PyG's existing increment rule for keys containing "index".
                     root_id=torch.from_numpy(roots.copy()),center_index=torch.from_numpy(centers+ptr[:-1]),
                     batch=torch.from_numpy(np.repeat(np.arange(len(indices)),sizes)),ptr=torch.from_numpy(ptr))
        result.y=result.y_levels[:,-1]
        if geometry_required:
            result.pos=torch.from_numpy(pos);result.rel_pos=torch.from_numpy(rel);result.pos_enc=torch.from_numpy(lpe)
        if synapses is not None:result.synapse_id=torch.from_numpy(synapses)
        if post is not None:result.postsynaptic_embedding=torch.from_numpy(post)
        return result
