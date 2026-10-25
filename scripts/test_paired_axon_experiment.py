"""Scientific invariants for the paired training experiment."""
import json
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts import paired_axon_experiment as prep
from scripts import train_paired_axon_experiment as train
from gnn.model import ModelConfig,WindowClassifier
from data.dataset_lcpn import load_hierarchy


def test_ragged_pooling_matches_ordinary_model():
    manifest=json.loads((prep.SOURCE/'fold0/manifest.json').read_text())
    hierarchy=load_hierarchy(manifest);torch.manual_seed(1)
    model=WindowClassifier(ModelConfig(architecture='pointwise_mlp',postsynaptic_dim=64),hierarchy).eval()
    counts=np.array([1,3,10]);offsets=np.r_[0,np.cumsum(counts)];raw=np.random.default_rng(1).normal(size=(14,64)).astype(np.float32)
    x,mask=prep.padded(raw,offsets,np.arange(14),np.arange(3))
    x=torch.from_numpy(x);mask=torch.from_numpy(mask);aux=torch.randn(3,64)
    bag=train.hidden(model,x,mask,aux)
    ordinary=model(x[mask],torch.empty((2,0),dtype=torch.long),torch.repeat_interleave(torch.arange(3),torch.from_numpy(counts)),postsynaptic_embedding=aux)
    torch.testing.assert_close(bag,ordinary)
    paths=list(hierarchy.label_paths.values())[:3]
    targets=torch.tensor([[hierarchy.level_maps[i][v] for i,v in enumerate(p)] for p in paths])
    loss=model.cls_head.compute_loss(bag,targets);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_matched_balanced_budget_and_determinism():
    class D:
        def __init__(self,multiplier):
            self.labels=np.repeat(np.arange(6),np.arange(1,7)*multiplier if multiplier==50 else np.full(6,multiplier));self.roots=list(range(6))
            self.index=np.column_stack([self.labels,np.zeros(len(self.labels),dtype=np.int32),np.arange(len(self.labels))])
        def __len__(self):return len(self.labels)
    label_counts=[]
    for multiplier in [10,50]:
        sampler=train.BalancedBatches(D(multiplier),budget=103,batch_size=32,seed=7,reference_counts=np.arange(1,7)*10)
        first=list(sampler);second=list(sampler)
        label_counts.append(np.bincount(sampler.ds.labels[np.concatenate(first)],minlength=6))
        assert first==second and len(first)==4 and sum(map(len,first))==103
        sampler.epoch=1;assert list(sampler)!=first
    np.testing.assert_array_equal(*label_counts)


def test_inner_split_never_uses_outer_test_roots(tmp_path):
    cells=[];manifest={'cells':{}}
    for label in range(6):
        for j in range(4):
            root=str(label*4+j);cells.append({'root_id':root,'targets':[0,0,label]})
            manifest['cells'][root]={'split':'test' if j==3 else 'train'}
            p=tmp_path/'cells'/root;p.mkdir(parents=True)
            (p/'complete.json').write_text(json.dumps({'axon_by_fold':[2]*5}))
    with mock.patch.object(prep,'BASE',tmp_path):
        splits=prep.split_roots(manifest,{'cells':cells},0)
    assert len(splits['validation'])==6 and len(splits['train'])==12 and len(splits['test'])==6
    assert not(set(splits['train'])&set(splits['validation']) or set(splits['train'])&set(splits['test']) or set(splits['validation'])&set(splits['test']))
    assert splits['test']==[3,7,11,15,19,23]


def test_mixed_input_routing_and_bulk_loader(tmp_path):
    folder=tmp_path/'cells'/'123';folder.mkdir(parents=True)
    for kind in ['axon','dendrite']:
        x=np.full((2,10,64),3,np.float32);mask=np.ones((2,10),bool)
        aux=np.full((2,64),7 if kind=='axon' else 3,np.float32)
        for name,value in dict(x=x,mask=mask,aux=aux).items():np.save(folder/f'{kind}_{name}.npy',value)
    np.save(folder/'axon_synapse_id.npy',np.array([100,101]))
    (tmp_path/'indices').mkdir();np.save(tmp_path/'indices/fold0_train_axon_dendrite.npy',np.array([[0,0,1],[0,1,0]],np.int32))
    plan={'cells':[{'root_id':'123','targets':[0,1,2]}]}
    with mock.patch.object(train,'BASE',tmp_path):
        dataset=train.Bags(plan,0,'train','axon_dendrite')
        loader=DataLoader(dataset,batch_sampler=[[0,1]],collate_fn=train.identity,num_workers=0)
        x,mask,aux,targets,roots,syn=next(iter(loader))
    assert x.shape==(2,10,64)
    assert torch.all(aux[0]==7) and torch.all(aux[1]==x[1].mean(0))
    assert syn.tolist()==[101,-1] and roots.tolist()==[123,123]


def test_evaluate_decodes_bfloat16_features_in_float32():
    manifest=json.loads((prep.SOURCE/'fold0/manifest.json').read_text())
    hierarchy=load_hierarchy(manifest)
    model=WindowClassifier(ModelConfig(architecture='pointwise_mlp',postsynaptic_dim=64),hierarchy).eval()
    path=next(iter(hierarchy.label_paths.values()))
    targets=torch.tensor([[hierarchy.level_maps[i][v] for i,v in enumerate(path)]])
    batch=(torch.randn(1,10,64),torch.ones(1,10,dtype=torch.bool),torch.randn(1,64),targets,torch.tensor([1]),torch.tensor([2]))
    original=train.hidden
    def reduced_precision(*args):
        return original(*args).to(torch.bfloat16)
    with mock.patch.object(train,'hidden',side_effect=reduced_precision):
        result=train.evaluate(model,[batch],torch.device('cpu'),hierarchy.level_classes[-1])
    assert 'window' in result and 'cell' in result
