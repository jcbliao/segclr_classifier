"""Whole-window packing, sampling and default behavior across training entry points."""
import copy
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from data.dataset_windowed import WindowedGraphDatasetLCPN, balanced_sampler
from data.packed_windows import PackedWholeCellWindows
from scripts.test_batched_presynaptic import fixture


def whole_windows(thickness=True):
    _, hierarchy = fixture()
    ds = WindowedGraphDatasetLCPN.__new__(WindowedGraphDatasetLCPN)
    ds.hierarchy = hierarchy
    ds.classes = hierarchy.level_classes[-1]
    ds.num_embeddings = 10
    ds.pos_dim = 8
    ds.cell_data = {}
    ds.cell_membership = {}
    ds.index_root_ids = np.array([11, 11, 11, 22, 22, 22])
    ds.index_centers = np.tile(np.arange(3), 2)
    ds.index_membership_centers = ds.index_centers.copy()
    ds.index_labels = np.repeat([0, 1], 3)
    template, _ = fixture()
    labels = template.__getitems__([0, 3]).y_levels
    for k, root in enumerate([11, 22]):
        graph = Data(x=torch.arange(192).reshape(3,64).float()/100,
                     pos=torch.arange(9).reshape(3,3).float()*100,
                     edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                     edge_attr=torch.ones(4,1),
                     y_levels=labels[k:k+1],root_id=root)
        if thickness:
            graph.thickness = torch.arange(6).reshape(3,2).float()
        ds.cell_data[root] = graph
        ds.cell_membership[root] = (np.array([0,2,5,6]),np.array([0,1,1,0,2,2]))
    return ds


@pytest.mark.parametrize('thickness',[False,True])
def test_whole_window_packing_preserves_every_field_and_repeated_draws(thickness):
    source = whole_windows(thickness)
    packed = PackedWholeCellWindows(source)
    assert packed.prepare_packed(workers=2)
    indices = [4,0,2,0,5,1]
    expected = Batch.from_data_list([source[i] for i in indices])
    actual = packed.get_batch(indices)
    for key in expected.keys():
        torch.testing.assert_close(actual[key], expected[key],rtol=0,atol=0)
    np.testing.assert_array_equal(packed.window_sizes,[2,3,1,2,3,1])
    assert packed.cell_data is source.cell_data
    assert packed.index_centers is source.index_centers
    # Balanced draws use the exact same labels and seeded Torch RNG.
    torch.manual_seed(7)
    original = list(balanced_sampler(source))
    torch.manual_seed(7)
    assert original == list(balanced_sampler(packed))


def test_whole_window_memory_limit_leaves_original_streaming_data_usable():
    packed = PackedWholeCellWindows(whole_windows())
    with pytest.raises(MemoryError):
        packed.prepare_packed(max_bytes=1)
    assert not hasattr(packed,'_packed')
    assert packed.get_batch([1,0]).num_graphs == 2


def test_shared_cli_enables_fast_defaults_and_honors_explicit_overrides(tmp_path):
    from scripts.train_gnn import build_training_parser
    from scripts.training_config import parse_training_args
    args = parse_training_args(build_training_parser(),[])
    assert args.amp and args.optimized_loading and args.cuda_graphs
    args = parse_training_args(build_training_parser(),['--no-amp','--no-optimized-loading','--no-cuda-graphs'])
    assert not (args.amp or args.optimized_loading or args.cuda_graphs)
    path = tmp_path/'config.json'
    path.write_text('{"amp": false, "cuda_graphs": false, "optimized_loading": false}')
    args = parse_training_args(build_training_parser(),['--config',str(path),'--amp'])
    assert args.amp and not args.cuda_graphs and not args.optimized_loading


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
@pytest.mark.parametrize('architecture',['mean','linear','pointwise_mlp','graph_transformer','graph_transformer_teasar'])
def test_whole_window_graph_updates_match_original_windows(architecture):
    from gnn.model import WindowClassifier,ModelConfig
    from gnn.window_graph_training import WindowGraphTrainer
    from gnn.training_performance import prepare_routes,move_routes,routed_loss
    source = whole_windows()
    ds = PackedWholeCellWindows(source)
    ds.prepare_packed(workers=2)
    ds.prepare_device(torch.device('cuda'))
    config = ModelConfig(architecture=architecture,gt_use_thickness=architecture.startswith('graph_transformer'),
                         use_position=architecture=='mean',use_lpe=architecture=='mean')
    torch.manual_seed(21)
    model = WindowClassifier(config,source.hierarchy).cuda().train()
    reference = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(),lr=1e-4,fused=True)
    refopt = torch.optim.Adam(reference.parameters(),lr=1e-4,fused=True)
    runner = WindowGraphTrainer(model,opt,torch.amp.GradScaler('cuda',enabled=False),ds,amp=False,bucket_size=4)
    for indices in [[0,2,4],[1,3],[1,3],[0,2,4]]:
        cpu = Batch.from_data_list([source[i] for i in indices])
        routes = move_routes(prepare_routes(reference.cls_head,cpu.y_levels),torch.device('cuda'))
        b = cpu.to('cuda')
        refopt.zero_grad(set_to_none=True)
        hidden = reference(b.x,b.edge_index,b.batch,pos_enc=b.pos_enc,rel_pos=b.rel_pos,thickness=b.thickness,
                           has_segclr=torch.ones(b.num_nodes,device='cuda',dtype=torch.bool),
                           batch_shape=(len(indices),int(ds.window_sizes[indices].max())))
        routed_loss(reference.cls_head,hidden,b.y_levels,routes).backward()
        refopt.step()
        runner.step_indices(indices)
        torch.cuda.synchronize()
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-4,atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
def test_tuple_prefetch_preserves_data_and_cleans_up():
    from gnn.cuda_prefetch import CUDATuplePrefetcher
    source = [(torch.arange(4),torch.ones(4,2)) for _ in range(3)]
    for actual,expected in zip(CUDATuplePrefetcher(source,torch.device('cuda')),source):
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a.cpu(),e)
    for batch in CUDATuplePrefetcher(source,torch.device('cuda')):
        break


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
@pytest.mark.parametrize('architecture,classifier,graphs',[
    ('mean','lcpn',True),('linear','lcpn',True),('pointwise_mlp','lcpn',True),
    ('graph_transformer','lcpn',True),('graph_transformer_teasar','lcpn',True),
    ('mpnn','lcpn',False),('mpnn_complete','lcpn',False),('mean','flat',False),
    ('graph_transformer','lcpn',False)])
def test_real_whole_cell_trainer_uses_defaults_and_resumes(tmp_path,monkeypatch,architecture,classifier,graphs):
    import json
    import scripts.train_gnn as trainer
    from scripts.training_config import parse_training_args
    monkeypatch.setattr(trainer,'WindowedGraphDatasetLCPN',lambda *a,**kw:whole_windows(kw['use_thickness']))
    monkeypatch.setattr(trainer,'publish_test_prediction_cache',lambda *a,**kw:None)
    args = parse_training_args(trainer.build_training_parser(),[
        '--dataset','all_windows','--architecture',architecture,'--classifier',classifier,
        '--num-embeddings','10','--batch-size','2','--epochs','1','--num-workers','0',
        '--results-dir',str(tmp_path),'--manifest',
        'analysis/presynaptic/cave_teasar_confidence_5fold_20261010/performance/smoke_manifest.json'])
    if architecture=='graph_transformer' and not graphs:
        args.gt_dropout = .1
    trainer.main(args)
    args.resume = True
    args.epochs = 2
    trainer.main(args)
    paths = list(tmp_path.glob('*/performance.jsonl'))
    assert len(paths) == 1
    rows = [json.loads(s) for s in paths[0].read_text().splitlines()]
    assert [r['epoch'] for r in rows] == [0,1]
    assert all(r['resident_features'] and r['amp_dtype']=='float16' and r['cuda_graphs']==graphs for r in rows)
    assert all(r['windows']==6 and r['optimizer_steps']==3 for r in rows)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA allocation required')
def test_paired_trainer_resident_data_fp16_and_checkpoint_resume(tmp_path,monkeypatch):
    import json
    import scripts.train_paired_axon_experiment as paired
    _,h=fixture()
    cells=[]
    for leaf in h.level_classes[-1][:6]:
        name,path=next((name,path) for name,path in h.label_paths.items() if path[-1]==leaf)
        targets=[h.level_maps[i][part] for i,part in enumerate(path)]
        cells.append({'root_id':str(11+len(cells)),'targets':targets})
    plan={'seed':0,'batch_size':4,'cells':cells,'level_classes':h.level_classes}
    (tmp_path/'indices').mkdir()
    records=np.array([[root,kind,0] for root in range(6) for kind in [0,1]],np.int64)
    for suffix in ['train_axon_only','validation_shared','test_shared']:
        np.save(tmp_path/'indices'/f'fold0_{suffix}.npy',records)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'complete.json').write_text(json.dumps({'plan_sha256':paired.sha(tmp_path/'plan.json')}))
    (tmp_path/'fold0').mkdir()
    (tmp_path/'fold0/manifest.json').write_text('{}')
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/paired_axon_experiment.py').write_text('# isolated test preparation')
    for cell in cells:
        folder=tmp_path/'cells'/cell['root_id'];folder.mkdir(parents=True)
        for kind in ['axon','dendrite']:
            np.save(folder/f'{kind}_x.npy',np.arange(640,dtype=np.float32).reshape(1,10,64)/100)
            mask=np.ones((1,10),bool);mask[:,5:]=False
            np.save(folder/f'{kind}_mask.npy',mask)
            np.save(folder/f'{kind}_aux.npy',np.ones((1,64),np.float32))
        np.save(folder/'axon_synapse_id.npy',np.array([123],np.int64))
    monkeypatch.setattr(paired,'BASE',tmp_path)
    monkeypatch.setattr(paired,'SOURCE',tmp_path)
    monkeypatch.setattr(paired,'ROOT',tmp_path)
    monkeypatch.setattr(paired,'frozen_load',lambda:plan)
    monkeypatch.setattr(paired,'load_hierarchy',lambda _:h)
    bags=paired.Bags(plan,0,'train','axon_only')
    indices=[4,0,2,0]
    expected=bags.__getitems__(indices)
    assert bags.prepare_device(workers=2)
    for a,b in zip(bags.__getitems__(indices),expected):
        torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0)
    original_checkpoint=paired.checkpoint
    def interrupt(path,state):
        original_checkpoint(path,state)
        if path.name=='checkpoint_last.pt' and state['epoch']==0:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(paired,'checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        paired.train(0,'axon_only',2,2)
    monkeypatch.setattr(paired,'checkpoint',original_checkpoint)
    paired.train(0,'axon_only',2,2)
    checkpoint=next(tmp_path.glob('results/**/checkpoint_last.pt'))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==1 and state['amp_dtype']=='float16' and state['grad_scaler_state']
    assert all(group['fused'] for group in state['optimizer_state']['param_groups'])
    assert all(torch.isfinite(value).all() for value in state['model_state'].values())
