"""Interventions preserve baseline computation and window boundaries."""
from types import SimpleNamespace
import pytest
import torch
from torch_geometric.data import Data,Batch
from gnn.graph_transformer import GraphAttention
from gnn.structure_inspection import intervention,randomized_edges
from gnn.structure_statistics import StructureStatistics


def attention_fixture():
    torch.manual_seed(7)
    attn=GraphAttention(8,num_heads=2).eval()
    model=SimpleNamespace(graph_transformer=SimpleNamespace(blocks=[SimpleNamespace(attn=attn)]),encoder=None)
    x=torch.randn(2,5,8)
    valid=torch.tensor([[True]*5,[True]*3+[False]*2])
    adj=torch.eye(5).repeat(2,1,1);adj[:,1,2]=adj[:,2,1]=1
    return model,attn,x,adj,valid


def test_identity_and_restoration():
    model,attn,x,adj,valid=attention_fixture();expected=attn(x,adj,valid)
    original=attn.forward
    with intervention(model,'identity'):
        torch.testing.assert_close(attn(x,adj,valid),expected)
    assert attn.forward==original
    with pytest.raises(ValueError):
        with intervention(model,'no_qk'):raise ValueError('test')
    assert attn.forward==original


def test_uniform_mask_and_nodewise_readout():
    model,attn,x,adj,valid=attention_fixture()
    class Observer:
        def attention(self,layer,content,topology,weights,*rest):self.weights=weights
    observer=Observer()
    with intervention(model,'uniform_attention',observer):attn(x,adj,valid)
    weights=observer.weights
    torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights.sum(-1)))
    assert torch.count_nonzero(weights[1,:,:,3:])==0
    with intervention(model,'nodewise_mean_readout',observer):out=attn(x,adj,valid)
    assert torch.count_nonzero(out[:,1:])==0
    assert torch.count_nonzero(observer.weights[:,:,0,0])==0
    torch.testing.assert_close(observer.weights[:,:,0].sum(-1),torch.ones(2,2))
    assert torch.count_nonzero(out[:,0])>0


def test_rewire_preserves_simple_graphs_counts_and_windows():
    batch=Batch.from_data_list([
        Data(x=torch.randn(4,3),edge_index=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])),
        Data(x=torch.randn(2,3),edge_index=torch.tensor([[0,1],[1,0]])),
        Data(x=torch.randn(1,3),edge_index=torch.empty(2,0,dtype=torch.long))])
    a=randomized_edges(batch,torch.Generator().manual_seed(4))
    b=randomized_edges(batch,torch.Generator().manual_seed(4))
    assert torch.equal(a,b)
    pairs=set(map(tuple,a.t().tolist()))
    assert len(pairs)==a.shape[1]==batch.edge_index.shape[1]
    assert all(s!=t and (t,s) in pairs and batch.batch[s]==batch.batch[t] for s,t in pairs)
    assert torch.equal(torch.bincount(batch.batch[a[0]],minlength=3),torch.tensor([6,2,0]))


def test_observation_finite_and_mass_partition():
    model,attn,x,adj,valid=attention_fixture();observer=StructureStatistics()
    with intervention(model,'identity',observer):attn(x,adj,valid)
    stats=observer.export()
    for head in range(2):
        prefix=f'GT/layer0/head{head}/node/'
        total=sum(stats[prefix+k+'_mass']['mean'] for k in ['self','neighbor','other_node','cls_key'])
        assert total==pytest.approx(1,abs=1e-6)
    assert all(torch.isfinite(torch.tensor(row['mean'])) for row in stats.values())


def test_mpnn_branch_removal_restored():
    from torch_geometric.nn import SAGEConv
    conv=SAGEConv(3,4)
    model=SimpleNamespace(graph_transformer=None,encoder=SimpleNamespace(convs=[conv]))
    x=torch.randn(3,3);edges=torch.tensor([[0,1,1,2],[1,0,2,1]])
    baseline=conv(x,edges)
    with intervention(model,'no_neighbors'):
        torch.testing.assert_close(conv(x,edges),conv.lin_r(x))
    with intervention(model,'no_self'):
        removed=conv(x,edges)
    torch.testing.assert_close(conv(x,edges),baseline)
    torch.testing.assert_close(removed+conv.lin_r(x),baseline)
