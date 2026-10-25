"""Batch equivalence, ragged topology, missing embeddings and gradient checks."""
from collections import OrderedDict
import copy
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from data.batched_presynaptic import BatchedPresynapticWindowDataset,collate_presynaptic
from data.dataset_presynaptic import PresynapticWindowDataset
from data.dataset_lcpn import load_hierarchy
from gnn.model import ModelConfig,WindowClassifier


def fixture(precomputed=False,single=False,post=False):
    ds=BatchedPresynapticWindowDataset.__new__(BatchedPresynapticWindowDataset)
    h=load_hierarchy();path=next(iter(h.label_paths.values()))
    target=torch.tensor([[h.level_maps[i][v] for i,v in enumerate(path)]])
    ds.index_root_ids=np.array([11,11,22,22]);ds.index_window_rows=np.array([0,1,0,1]);ds.window_sizes=np.array([3,2,3,2]) if not single else np.ones(4,int)
    ds.embedding_augmentation=None;ds.single_presynaptic_embedding=single;ds.postsynaptic_cache='fixture' if post else None;ds.use_postsynaptic=post
    cells={}
    for rid in [11,22]:
        c=dict(pos=np.arange(15,dtype=np.float64).reshape(5,3)+rid,observed_ids=np.array([4,0,2]),observed_x=np.arange(192,dtype=np.float32).reshape(3,64)+rid,
               centers=np.array([0,4]),offsets=np.array([0,3,5]),members=np.array([0,1,2,2,4]),lpe=np.arange(40,dtype=np.float32).reshape(5,8),
               targets=target,observed_lookup={4:0,0:1,2:2},adjacency=[[1],[0,2],[1,4],[],[2]])
        if precomputed:c.update(window_edge_offsets=np.array([0,4,6]),window_edges=np.array([[0,1],[1,0],[1,2],[2,1],[0,1],[1,0]]))
        if post:c.update(synapse_ids=np.array([rid*100,rid*100+1]),post_x=np.ones((2,64),np.float32)*rid)
        cells[rid]=c
    ds._load_cell=lambda rid:cells[rid]
    return ds,h


def same(a,b):
    assert set(a.keys())==set(b.keys())
    for k in a.keys():torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert a.num_graphs==b.num_graphs


@pytest.mark.parametrize('precomputed',[False,True])
@pytest.mark.parametrize('single',[False,True])
@pytest.mark.parametrize('post',[False,True])
def test_batch_exact(precomputed,single,post):
    ds,_=fixture(precomputed,single,post);indices=[0,2,1,0,3,2]
    reference=Batch.from_data_list([ds[i] for i in indices]);bulk=ds.__getitems__(indices)
    same(reference,bulk)
    loader=DataLoader(ds,batch_sampler=[indices],collate_fn=collate_presynaptic)
    same(reference,next(iter(loader)))


@pytest.mark.parametrize('architecture',['mean','pointwise_mlp','graph_transformer'])
def test_forward_and_gradients_match(architecture):
    ds,h=fixture();idx=[0,2,1,0,3]
    old=Batch.from_data_list([ds[i] for i in idx]);new=ds.__getitems__(idx)
    torch.manual_seed(12);m=WindowClassifier(ModelConfig(architecture=architecture),h).eval();other=copy.deepcopy(m)
    def step(model,b):
        out=model(b.x,b.edge_index,b.batch,pos_enc=b.pos_enc,rel_pos=b.rel_pos)
        loss=model.cls_head.compute_loss(out,b.y_levels);loss.backward();return out,loss
    a,la=step(m,old);b,lb=step(other,new)
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(la,lb,rtol=0,atol=0)
    for p,q in zip(m.parameters(),other.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_augmentation_preserves_draw_order():
    ds,_=fixture();ds.embedding_augmentation='flip'
    for rid in [11,22]:
        c=ds._load_cell(rid);c['augmentation_lookup']={4:0,0:1,2:2};c['augmentation_choices']=np.arange(3*5*64,dtype=np.float32).reshape(3,5,64)
    idx=[2,0,3,2,1];torch.manual_seed(32);a=Batch.from_data_list([ds[i] for i in idx]);torch.manual_seed(32);b=ds.__getitems__(idx);same(a,b)
