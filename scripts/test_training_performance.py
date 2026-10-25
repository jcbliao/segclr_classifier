"""Performance paths must preserve batches, objectives, gradients and decisions."""
import copy
import numpy as np
import pytest
import torch
from data.dataset_lcpn import load_hierarchy
from data.mixed_cell_batch_sampler import MixedCellBatchSampler
from gnn.model import ModelConfig,WindowClassifier
from gnn.training_performance import prepare_routes,move_routes,routed_loss,static_top_down
from gnn.metrics import majority_vote_by_group
from scripts.profile_training_overheads import reference_sampler,legacy_vote
from scripts.test_batched_presynaptic import fixture


@pytest.mark.parametrize('balance',[False,True])
@pytest.mark.parametrize('epoch',[0,1,7])
def test_sampler_preserves_every_draw_and_batch(balance,epoch):
    class Dataset:
        index_root_ids=np.random.default_rng(31).integers(0,24,5000)
        index_labels=index_root_ids%6
        window_sizes=np.random.default_rng(33).integers(1,18,5000)
        def __len__(self):return len(self.index_labels)
    ds=Dataset();kw=dict(cells_per_batch=16,attention_budget=10000000,max_windows=64,shuffle=True,seed=5,balance_classes=balance)
    old=reference_sampler()(ds,**kw);new=MixedCellBatchSampler(ds,**kw);old.set_epoch(epoch);new.set_epoch(epoch)
    assert list(old)==list(new)


@pytest.mark.parametrize('weighted',[False,True])
@pytest.mark.parametrize('single_class',[False,True])
def test_routed_loss_preserves_gradients_and_unused_heads(weighted,single_class):
    h=load_hierarchy();torch.manual_seed(4);model=WindowClassifier(ModelConfig(architecture='mean'),h);head=model.cls_head
    if weighted:head.set_class_weights({i:torch.arange(1,n['n_children']+1).float() for i,n in enumerate(head.nodes)})
    other=copy.deepcopy(head);paths=list(h.label_paths.values());selected=[paths[0] if single_class else paths[i%len(paths)] for i in range(32)]
    y=torch.tensor([[h.level_maps[i][v] for i,v in enumerate(p)] for p in selected]);x=torch.randn(32,64,requires_grad=True);z=x.detach().clone().requires_grad_()
    a=head.compute_loss(x,y);b=routed_loss(other,z,y,move_routes(prepare_routes(other,y),torch.device('cpu')))
    a.backward();b.backward();torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(x.grad,z.grad,rtol=0,atol=0)
    for p,q in zip(head.parameters(),other.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


@pytest.mark.parametrize('ties',[False,True])
def test_static_routing_predictions(ties):
    h=load_hierarchy();torch.manual_seed(2);head=WindowClassifier(ModelConfig(architecture='mean'),h).eval().cls_head
    if ties:
        for p in head.parameters():p.data.zero_()
    x=torch.randn(23,64);assert torch.equal(head.predict_top_down(x),static_top_down(head,x))


@pytest.mark.parametrize('arch',['mean','pointwise_mlp','graph_transformer','mpnn'])
def test_known_batch_shape_and_unused_geometry(arch):
    ds,h=fixture();full=ds.__getitems__([0,2,1,0]);ds.geometry_required=arch not in {'mean','pointwise_mlp'};fast=ds.__getitems__([0,2,1,0])
    torch.manual_seed(2);m=WindowClassifier(ModelConfig(architecture=arch),h).eval();other=copy.deepcopy(m)
    def run(model,b,shape):
        out=model(b.x,b.edge_index,b.batch,pos_enc=getattr(b,'pos_enc',None),rel_pos=getattr(b,'rel_pos',None),batch_shape=shape)
        loss=model.cls_head.compute_loss(out,b.y_levels);loss.backward();return out
    a=run(m,full,None);b=run(other,fast,(4,3));torch.testing.assert_close(a,b,rtol=0,atol=0)
    for p,q in zip(m.parameters(),other.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


@pytest.mark.parametrize('size',[0,1,1000])
def test_vote_exact_including_ties_and_order(size):
    rng=np.random.default_rng(2);roots=rng.integers(0,17,size);true=rng.integers(0,6,size).astype(np.int16);pred=rng.integers(0,6,size).astype(np.int16)
    old=legacy_vote(roots,true,pred);new=majority_vote_by_group(roots,true,pred)
    for a,b in zip(old,new):assert np.array_equal(a,b) and a.dtype==b.dtype

@pytest.mark.parametrize('architecture',['mean','pointwise_mlp','graph_transformer'])
def test_trainer_evaluation_preserves_labels_roots_and_predictions(architecture):
    from scripts.train_gnn import evaluate
    from data.batched_presynaptic import collate_presynaptic
    from torch.utils.data import DataLoader
    ds,h=fixture();torch.manual_seed(9)
    model=WindowClassifier(ModelConfig(architecture=architecture),h).eval()
    full=ds.__getitems__([0,2,1,3])
    with torch.no_grad():
        hidden=model(full.x,full.edge_index,full.batch,pos_enc=full.pos_enc,rel_pos=full.rel_pos)
        expected=model.cls_head.predict_top_down(hidden)[:,-1].numpy()
    ds.geometry_required=architecture=='graph_transformer'
    loader=DataLoader(ds,batch_sampler=[[0,2,1,3]],collate_fn=collate_presynaptic)
    truth,pred,roots=evaluate(model,loader,torch.device('cpu'))
    np.testing.assert_array_equal(pred,expected)
    np.testing.assert_array_equal(truth,full.y_levels[:,-1].numpy())
    np.testing.assert_array_equal(roots,full.root_id.numpy())

@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU validation requires allocation')
@pytest.mark.parametrize('architecture',['mean','pointwise_mlp','graph_transformer'])
def test_cuda_adam_step_preserves_objective_and_parameters(architecture):
    ds,h=fixture();batch=ds.__getitems__([0,2,1,3]);shape=(4,3)
    torch.manual_seed(10);model=WindowClassifier(ModelConfig(architecture=architecture),h).cuda().eval();other=copy.deepcopy(model)
    routes=move_routes(prepare_routes(other.cls_head,batch.y_levels),torch.device('cuda'));batch=batch.to('cuda')
    opts=[torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5,fused=i==1) for i,m in enumerate([model,other])]
    losses=[]
    for m,opt,fast in zip([model,other],opts,[False,True]):
        hidden=m(batch.x,batch.edge_index,batch.batch,pos_enc=batch.pos_enc,rel_pos=batch.rel_pos,batch_shape=shape if fast else None)
        loss=routed_loss(m.cls_head,hidden,batch.y_levels,routes) if fast else m.cls_head.compute_loss(hidden,batch.y_levels)
        loss.backward();losses.append(loss)
    torch.testing.assert_close(*losses,rtol=1e-5,atol=1e-6)
    for a,b in zip(model.parameters(),other.parameters()):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-5)
    for opt in opts:opt.step()
    for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

@pytest.mark.parametrize('weighted',[False,True])
@pytest.mark.parametrize('single_class',[False,True])
def test_dense_routing_preserves_loss_gradients_and_unused_heads(weighted,single_class):
    from gnn.training_performance import prepare_dense_routes,dense_routed_loss
    h=load_hierarchy();torch.manual_seed(17);head=WindowClassifier(ModelConfig(architecture='mean'),h).cls_head
    if weighted:head.set_class_weights({i:torch.arange(1,n['n_children']+1).float() for i,n in enumerate(head.nodes)})
    other=copy.deepcopy(head);paths=list(h.label_paths.values());selected=[paths[0] if single_class else paths[i%len(paths)] for i in range(32)]
    y=torch.tensor([[h.level_maps[i][v] for i,v in enumerate(p)] for p in selected]);x=torch.randn(32,64,requires_grad=True);z=x.detach().clone().requires_grad_()
    routes,count=prepare_dense_routes(other,y);expected=head.compute_loss(x,y);actual=dense_routed_loss(other,z,routes,count)
    expected.backward();actual.backward();torch.testing.assert_close(expected,actual,rtol=1e-6,atol=1e-7);torch.testing.assert_close(x.grad,z.grad,rtol=1e-5,atol=1e-7)
    for p,q in zip(head.parameters(),other.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7)

@pytest.mark.parametrize('mixed',[False,True])
@pytest.mark.parametrize('parallel',[False,True])
def test_precomputed_epoch_plans_preserve_all_seeded_draws(mixed,parallel):
    from data.epoch_plans import SamplingData,precompute_epoch_plans
    from data.dataset_presynaptic import AttentionBudgetBatchSampler
    rng=np.random.default_rng(32);roots=rng.integers(0,24,5000)
    ds=SamplingData(roots,roots%6,rng.integers(1,18,5000))
    kwargs=dict(attention_budget=10000000,max_windows=64,shuffle=True,seed=5,balance_classes=True)
    if mixed:kwargs['cells_per_batch']=16
    sampler=(MixedCellBatchSampler if mixed else AttentionBudgetBatchSampler)(ds,**kwargs)
    plans=precompute_epoch_plans(sampler,1,4,workers=2,parallel_threshold=0 if parallel else 100000)
    for epoch,plan in plans.items():
        sampler.set_epoch(epoch);reference=list(sampler)
        assert len(plan)==len(reference)
        for actual,expected in zip(plan,reference):np.testing.assert_array_equal(actual,expected)
