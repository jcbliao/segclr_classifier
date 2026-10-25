"""Packed windows preserve repeated draws, geometry, missing data and metadata."""
from collections import OrderedDict
import numpy as np
import pytest
import torch
from data.packed_presynaptic import PackedPresynapticWindowDataset
from scripts.test_batched_presynaptic import fixture,same

@pytest.mark.parametrize('geometry',[False,True])
@pytest.mark.parametrize('precomputed',[False,True])
@pytest.mark.parametrize('single',[False,True])
@pytest.mark.parametrize('post',[False,True])
def test_packed_batch_exact(geometry,precomputed,single,post):
    ds,h=fixture(precomputed,single,post);ds.hierarchy=h;ds._cache=OrderedDict();ds.geometry_required=geometry
    indices=[0,2,1,0,3,2];reference=ds.__getitems__(indices)
    ds.__class__=PackedPresynapticWindowDataset;assert ds.prepare_packed(workers=2)
    same(reference,ds.__getitems__(indices))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
def test_cuda_prefetch_preserves_batches_and_routes():
    from gnn.cuda_prefetch import CUDABatchPrefetcher
    from gnn.model import WindowClassifier,ModelConfig
    from torch.utils.data import DataLoader
    from data.batched_presynaptic import collate_presynaptic
    ds,h=fixture();model=WindowClassifier(ModelConfig(architecture='mean'),h).cuda()
    loader=DataLoader(ds,batch_sampler=[[0,2,1],[3,0,0]],collate_fn=collate_presynaptic)
    batches=list(CUDABatchPrefetcher(loader,torch.device('cuda'),model.cls_head));assert len(batches)==2
    for (gpu,shape,routes),indices in zip(batches,[[0,2,1],[3,0,0]]):
        cpu=ds.__getitems__(indices);same(cpu,gpu.cpu());assert shape==(3,3) and routes
    # Worker/stream cleanup also works when the consumer leaves early.
    for item in CUDABatchPrefetcher(loader,torch.device('cuda'),model.cls_head):break

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
def test_fused_adam_matches_update_and_skips_overflow():
    import copy
    torch.manual_seed(7)
    model=torch.nn.Linear(64,6).cuda();other=copy.deepcopy(model)
    x=torch.randn(128,64,device='cuda');target=torch.arange(128,device='cuda')%6
    for m,fused in [(model,False),(other,True)]:
        opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5,fused=fused)
        scaler=torch.amp.GradScaler('cuda',init_scale=256)
        with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(m(x),target)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        before=[p.detach().clone() for p in m.parameters()];scale=scaler.get_scale();opt.zero_grad(set_to_none=True)
        bad=sum(p.sum()*float('inf') for p in m.parameters())
        scaler.scale(bad).backward();scaler.step(opt);scaler.update()
        assert scaler.get_scale()<scale
        for p,q in zip(m.parameters(),before):torch.testing.assert_close(p,q,rtol=0,atol=0)
        state=scaler.state_dict();restored=torch.amp.GradScaler('cuda');restored.load_state_dict(state);assert restored.state_dict()==state
    for p,q in zip(model.parameters(),other.parameters()):torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
@pytest.mark.parametrize('architecture',['mean','pointwise_mlp','graph_transformer'])
def test_complete_cuda_graph_steps_match_repeated_adam_updates(architecture):
    import copy
    from gnn.model import WindowClassifier,ModelConfig
    from gnn.cuda_graph_training import CUDAGraphTrainer
    from gnn.training_performance import prepare_routes,prepare_dense_routes,move_routes,routed_loss
    ds,h=fixture();torch.manual_seed(4)
    model=WindowClassifier(ModelConfig(architecture=architecture),h).cuda().train();reference=copy.deepcopy(model)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5,fused=True)
    ref_opt=torch.optim.Adam(reference.parameters(),lr=1e-4,weight_decay=1e-5,fused=True)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    runner=CUDAGraphTrainer(model,opt,scaler,amp=False)
    for indices in [[0,2,1],[3,0],[0,2,1],[3,0],[0,2,1]]:
        cpu=ds.__getitems__(indices);shape=(len(indices),int((cpu.ptr[1:]-cpu.ptr[:-1]).max()))
        dense,count=prepare_dense_routes(model.cls_head,cpu.y_levels);dense=([(i,t.cuda()) for i,t in dense],count)
        routes=move_routes(prepare_routes(reference.cls_head,cpu.y_levels),torch.device('cuda'));b=cpu.to('cuda')
        ref_opt.zero_grad(set_to_none=True)
        hidden=reference(b.x,b.edge_index,b.batch,pos_enc=b.pos_enc,rel_pos=b.rel_pos,batch_shape=shape)
        routed_loss(reference.cls_head,hidden,b.y_levels,routes).backward();ref_opt.step()
        runner.step(b,shape,dense);torch.cuda.synchronize()
        for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=1e-4,atol=2e-6)
    assert runner.hits==3 and runner.misses==2

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
def test_complete_cuda_graph_fp16_scaler_and_overflow():
    from gnn.model import WindowClassifier,ModelConfig
    from gnn.cuda_graph_training import CUDAGraphTrainer
    from gnn.training_performance import prepare_dense_routes
    ds,h=fixture();torch.manual_seed(9);model=WindowClassifier(ModelConfig(architecture='mean'),h).cuda().train()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,fused=True);scaler=torch.amp.GradScaler('cuda',init_scale=256)
    runner=CUDAGraphTrainer(model,opt,scaler,amp=True)
    cpu=ds.__getitems__([0,2,1]);routes,count=prepare_dense_routes(model.cls_head,cpu.y_levels);routes=([(i,t.cuda()) for i,t in routes],count);b=cpu.to('cuda')
    runner.step(b,(3,3),routes);torch.cuda.synchronize();assert torch.isfinite(runner.loss_sum)
    before=[p.detach().clone() for p in model.parameters()];scale=scaler.get_scale()
    b.x.fill_(float('inf'));runner.step(b,(3,3),routes);torch.cuda.synchronize()
    assert scaler.get_scale()<scale and runner.hits==1
    for p,q in zip(model.parameters(),before):torch.testing.assert_close(p,q,rtol=0,atol=0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
@pytest.mark.parametrize('architecture',['mean','linear','pointwise_mlp','graph_transformer','graph_transformer_teasar'])
def test_masked_bucket_graph_preserves_repeated_updates_and_true_node_counts(architecture):
    import copy
    from gnn.model import WindowClassifier,ModelConfig
    from gnn.window_graph_training import WindowGraphTrainer
    from gnn.training_performance import prepare_routes,move_routes,routed_loss
    ds,h=fixture();ds.hierarchy=h;ds._cache=OrderedDict();ds.__class__=PackedPresynapticWindowDataset;ds.prepare_packed();ds.prepare_device(torch.device('cuda'))
    torch.manual_seed(21);m=WindowClassifier(ModelConfig(architecture=architecture),h).cuda().train();ref=copy.deepcopy(m)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5,fused=True);refopt=torch.optim.Adam(ref.parameters(),lr=1e-4,weight_decay=1e-5,fused=True)
    runner=WindowGraphTrainer(m,opt,torch.amp.GradScaler('cuda',enabled=False),ds,amp=False,bucket_size=4)
    for indices in [[0,2,1],[3,0],[3,0],[0,2,1],[0,2,1]]:
        cpu=ds.__getitems__(indices);shape=(len(indices),int((cpu.ptr[1:]-cpu.ptr[:-1]).max()))
        routes=move_routes(prepare_routes(ref.cls_head,cpu.y_levels),torch.device('cuda'));b=cpu.to('cuda');refopt.zero_grad(set_to_none=True)
        hidden=ref(b.x,b.edge_index,b.batch,pos_enc=b.pos_enc,rel_pos=b.rel_pos,has_segclr=b.has_segclr,batch_shape=shape);routed_loss(ref.cls_head,hidden,b.y_levels,routes).backward();refopt.step()
        runner.step_indices(indices);torch.cuda.synchronize()
        for p,q in zip(m.parameters(),ref.parameters()):torch.testing.assert_close(p,q,rtol=1e-4,atol=2e-6)
    assert runner.hits==3 and runner.misses==2

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
def test_masked_bucket_fp16_scale_growth_and_overflow_skip():
    from gnn.model import WindowClassifier,ModelConfig
    from gnn.window_graph_training import WindowGraphTrainer
    ds,h=fixture();ds.hierarchy=h;ds._cache=OrderedDict();ds.__class__=PackedPresynapticWindowDataset;ds.prepare_packed();ds.prepare_device(torch.device('cuda'))
    torch.manual_seed(7);m=WindowClassifier(ModelConfig(architecture='mean'),h).cuda().train()
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,fused=True);scaler=torch.amp.GradScaler('cuda',init_scale=128,growth_interval=2)
    runner=WindowGraphTrainer(m,opt,scaler,ds,bucket_size=4)
    runner.step_indices([0,2,1]);runner.step_indices([3,0]);torch.cuda.synchronize()
    assert scaler.get_scale()==256 and torch.isfinite(runner.loss_sum)
    before=[p.detach().clone() for p in m.parameters()]
    runner.fields['x'].fill_(float('inf'));runner.step_indices([3,0]);torch.cuda.synchronize()
    assert scaler.get_scale()==128 and runner.hits==1
    for p,q in zip(m.parameters(),before):torch.testing.assert_close(p,q,rtol=0,atol=0)
