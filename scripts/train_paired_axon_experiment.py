"""Train paired pointwise MLPs; select on validation axons, evaluate test once."""
from __future__ import annotations
import argparse,csv,json,math,os,sys,time
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.paired_axon_experiment import BASE,SOURCE,DOC,sha,atomic_json,frozen_load
from data.dataset_lcpn import load_hierarchy
from gnn.model import WindowClassifier,ModelConfig
from gnn.metrics import summarize,majority_vote_by_group
from gnn.cuda_prefetch import CUDATuplePrefetcher

class Bags(Dataset):
    def __init__(self,plan,fold,split,condition):
        suffix=condition if split=='train' else 'shared'
        self.index=np.load(BASE/'indices'/f'fold{fold}_{split}_{suffix}.npy',mmap_mode='r')
        self.roots=[c['root_id'] for c in plan['cells']]
        self.targets=np.asarray([c['targets'] for c in plan['cells']],np.int64)
        self.labels=self.targets[self.index[:,0],-1];self.cache=OrderedDict()
    def __len__(self):return len(self.index)
    def _arrays(self,root,kind):
        key=(root,kind)
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        folder=BASE/'cells'/self.roots[root];prefix='axon' if kind==0 else 'dendrite'
        arrays={k:np.load(folder/f'{prefix}_{k}.npy',mmap_mode='r') for k in ['x','mask','aux']}
        if kind==0:arrays['synapse_id']=np.load(folder/'axon_synapse_id.npy',mmap_mode='r')
        self.cache[key]=arrays
        while len(self.cache)>16:self.cache.popitem(last=False)
        return arrays
    def __getitems__(self,indices):
        if hasattr(self,'resident'):
            indices=np.asarray(indices,np.int64)
            device_ids=torch.from_numpy(indices).pin_memory().to('cuda',non_blocking=True)
            return tuple(t.index_select(0,device_ids) if t.is_cuda else t[indices] for t in self.resident)
        record=self.index[indices];n=len(record)
        x=np.empty((n,10,64),np.float32);mask=np.empty((n,10),bool);aux=np.empty((n,64),np.float32)
        syn=np.full(n,-1,np.int64)
        for code in np.unique(record[:,:2],axis=0):
            root,kind=map(int,code);where=np.flatnonzero((record[:,0]==root)&(record[:,1]==kind));rows=record[where,2]
            arrays=self._arrays(root,kind);x[where]=arrays['x'][rows];mask[where]=arrays['mask'][rows];aux[where]=arrays['aux'][rows]
            if kind==0:syn[where]=arrays['synapse_id'][rows]
        roots=np.asarray([int(self.roots[r]) for r in record[:,0]],np.int64)
        return tuple(torch.from_numpy(a) for a in [x,mask,aux,self.targets[record[:,0]].copy(),roots,syn])
    def __getitem__(self,index):return tuple(t[0] for t in self.__getitems__([index]))
    def prepare_device(self,workers=8):
        """Pack the exact bags once; leave labels/export metadata on the CPU."""
        sample=self.__getitems__([0]);required=sum(t[0].numel()*t.element_size()*len(self) for t in sample[:3])
        if required+8*2**30>torch.cuda.mem_get_info()[0] or required>44*2**30:
            return False
        packed=[torch.empty((len(self),*t.shape[1:]),dtype=t.dtype) for t in sample]
        def fill(start):
            local=copy.copy(self);local.cache=OrderedDict()
            stop=min(start+16384,len(self))
            values=local.__getitems__(np.arange(start,stop))
            for output,value in zip(packed,values):output[start:stop].copy_(value)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(fill,range(0,len(self),16384)))
            resident=tuple(t.cuda() if i<3 else t for i,t in enumerate(packed))
        except (MemoryError,torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache();return False
        self.resident=resident;self.cache.clear()
        print(f'Packed {len(self):,} paired bags; {required/2**30:.2f} GiB resident',flush=True)
        return True

def identity(batch):return batch

class BalancedBatches(Sampler):
    """Exact shared window/update budget, class-balanced draws, grouped I/O."""
    def __init__(self,dataset,budget,batch_size,seed,reference_counts=None):
        self.ds=dataset;self.budget=budget;self.batch_size=batch_size;self.seed=seed;self.epoch=0
        count=np.bincount(dataset.labels,minlength=6);assert (count>0).all()
        reference=count if reference_counts is None else np.asarray(reference_counts)
        assert (reference>0).all()
        self.class_probabilities=np.sqrt(reference)/np.sqrt(reference).sum()
        self.by_class=[np.flatnonzero(dataset.labels==j) for j in range(6)]
    def __len__(self):return math.ceil(self.budget/self.batch_size)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        classes=rng.choice(6,size=self.budget,replace=True,p=self.class_probabilities)
        selected=np.empty(self.budget,np.int64)
        for j in range(6):
            where=np.flatnonzero(classes==j)
            selected[where]=rng.choice(self.by_class[j],size=len(where),replace=True)
        # Mix up to 16 randomly ordered cells per I/O group without changing draws.
        root_order=rng.permutation(len(self.ds.roots));group=np.empty(len(root_order),np.int32)
        group[root_order]=np.arange(len(root_order))//16
        selected=selected[np.argsort(group[self.ds.index[selected,0]],kind='stable')]
        for start in range(0,len(selected),self.batch_size):yield selected[start:start+self.batch_size].tolist()

def hidden(model,x,mask,aux):
    transformed=model.pointwise_mlp(x)
    pooled=(transformed*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
    return torch.cat([pooled,aux.to(pooled.dtype)],dim=-1)

def metrics(true,pred,roots,classes):
    cell_true,cell_pred=majority_vote_by_group(roots,true,pred)
    return dict(window=summarize(true,pred,len(classes),classes),cell=summarize(cell_true,cell_pred,len(classes),classes))

@torch.inference_mode()
def evaluate(model,loader,device,classes,save=None):
    model.eval();ys=[];preds=[];roots=[];synapses=[];probs=[];logits=[[] for _ in range(model.cls_head.n_levels)]
    for x,mask,aux,y,root,syn in loader:
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            g=hidden(model,x.to(device,non_blocking=True),mask.to(device,non_blocking=True),aux.to(device,non_blocking=True))
        # Decode hierarchy probabilities in float32 after the mixed-precision encoder.
        decoded=model.cls_head.predict_distribution(g.float())
        ys.append(y[:,-1].numpy());preds.append(decoded['predictions'][:,-1].cpu().numpy());roots.append(root.numpy())
        if save:
            synapses.append(syn.numpy());probs.append(decoded['level_probabilities'][-1].float().cpu().numpy())
            for l,value in enumerate(decoded['level_logits']):logits[l].append(value.float().cpu().numpy())
    true,pred,root=map(np.concatenate,[ys,preds,roots])
    if save:
        payload=dict(target=true,prediction=pred,root_id=root,synapse_id=np.concatenate(synapses),
                     probabilities=np.concatenate(probs),classes=np.asarray(classes))
        payload.update({f'level{l}_logits':np.concatenate(values) for l,values in enumerate(logits)})
        tmp=Path(save).with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(save)
    return metrics(true,pred,root,classes)

def checkpoint(path,value):
    tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)

def train(fold,condition,workers,epochs):
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for production training')
    plan=frozen_load();digest=sha(BASE/'plan.json')
    assert json.loads((BASE/'complete.json').read_text())['plan_sha256']==digest
    seed=plan['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda');h=load_hierarchy(json.loads((SOURCE/f'fold{fold}/manifest.json').read_text()))
    assert h.level_classes==plan['level_classes'];classes=h.level_classes[-1]
    config=ModelConfig(architecture='pointwise_mlp',postsynaptic_dim=64,pointwise_mlp_layers=2,pointwise_mlp_hidden_dim=128)
    model=WindowClassifier(config,h).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,fused=True)
    scaler=torch.amp.GradScaler('cuda')
    datasets={s:Bags(plan,fold,s,condition) for s in ['train','validation','test']}
    for dataset in datasets.values():dataset.prepare_device(workers=min(8,max(1,workers)))
    budget=len(np.load(BASE/'indices'/f'fold{fold}_train_axon_only.npy',mmap_mode='r'))
    reference_index=np.load(BASE/'indices'/f'fold{fold}_train_axon_only.npy',mmap_mode='r')
    reference_counts=np.bincount(datasets['train'].targets[reference_index[:,0],-1],minlength=6)
    batch_size=plan['batch_size'];sampler=BalancedBatches(datasets['train'],budget,batch_size,seed,reference_counts)
    kwargs=dict(num_workers=workers,pin_memory=True,persistent_workers=workers>0,collate_fn=identity)
    if workers:kwargs.update(prefetch_factor=2)
    def loader_options(split):
        if not hasattr(datasets[split],'resident'):return kwargs
        options={**kwargs,'num_workers':0,'persistent_workers':False,'pin_memory':False}
        options.pop('prefetch_factor',None)
        return options
    loaders={'train':DataLoader(datasets['train'],batch_sampler=sampler,**loader_options('train'))}
    # Validation and final test workers are created only as needed; avoid 45 live workers.
    def eval_loader(split):
        return DataLoader(datasets[split],batch_sampler=[list(range(a,min(a+batch_size,len(datasets[split]))))
                          for a in range(0,len(datasets[split]),batch_size)],**{**loader_options(split),'persistent_workers':False})
    out=ROOT/'results/presynaptic/axon_vs_whole_cell_morpho6'/f'fold{fold}'/condition
    out.mkdir(parents=True,exist_ok=True);best=-1.;start=0
    identity_info=dict(plan_sha256=digest,fold=fold,condition=condition,config=asdict(config),
                       trainer_sha256=sha(__file__),preparation_code_sha256=sha(ROOT/'scripts/paired_axon_experiment.py'),
                       epochs=epochs,windows_per_epoch=budget,updates_per_epoch=len(sampler),
                       class_sampling_probabilities=sampler.class_probabilities.tolist(),
                       validation_index_sha256=sha(BASE/'indices'/f'fold{fold}_validation_shared.npy'),
                       test_index_sha256=sha(BASE/'indices'/f'fold{fold}_test_shared.npy'))
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['identity']!=identity_info:raise ValueError('Completed run differs')
        print('Already completed');return
    if (out/'checkpoint_last.pt').exists():
        ck=torch.load(out/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        assert ck['identity']==identity_info
        model.load_state_dict(ck['model_state']);optimizer.load_state_dict(ck['optimizer_state'])
        for group in optimizer.param_groups:group['fused']=True
        for state in optimizer.state.values():
            if 'step' in state:state['step']=state['step'].to(device)
        if ck.get('grad_scaler_state'):scaler.load_state_dict(ck['grad_scaler_state'])
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['epoch']+1;best=ck['best_validation_axon_macro_f1']
    history=[]
    history_path=out/'epoch_metrics.json'
    if history_path.exists():history=[r for r in json.loads(history_path.read_text()) if r['epoch']<start]
    plans={}
    if budget*max(0,epochs-start)*8<=512*2**20:
        def epoch_plan(epoch):
            local=copy.copy(sampler);local.epoch=epoch
            return epoch,[np.asarray(indices,np.int64) for indices in local]
        with ThreadPoolExecutor(max_workers=min(8,max(1,workers))) as pool:
            plans=dict(pool.map(epoch_plan,range(start,epochs)))
    for epoch in range(start,epochs):
        model.train();sampler.epoch=epoch;begin=time.monotonic();loss_sum=torch.zeros((),device=device,dtype=torch.float64);count=0;updates=0
        loader=(DataLoader(datasets['train'],batch_sampler=plans[epoch],**loader_options('train'))
                if epoch in plans else loaders['train'])
        for x,mask,aux,y,_,_ in CUDATuplePrefetcher(loader,device):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):loss=model.cls_head.compute_loss(hidden(model,x,mask,aux),y)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer);scaler.update()
            loss_sum.add_(loss.detach().double()*len(x));count+=len(x);updates+=1
        assert count==budget and updates==len(sampler)
        loss_sum=float(loss_sum)
        train_seconds=time.monotonic()-begin
        val=evaluate(model,eval_loader('validation'),device,classes)
        score=val['window']['macro_f1'];improved=score>best
        if improved:best=score
        state=dict(identity=identity_info,config=config,model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                   epoch=epoch,best_validation_axon_macro_f1=best,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                   grad_scaler_state=scaler.state_dict(),amp_dtype='float16')
        if improved:checkpoint(out/'checkpoint_best.pt',state)
        checkpoint(out/'checkpoint_last.pt',state)
        record=dict(epoch=epoch,train_loss=loss_sum/count,train_seconds=train_seconds,validation=val,
                    epoch_seconds=time.monotonic()-begin,windows=count,updates=updates)
        history.append(record);atomic_json(history_path,history);print(json.dumps(record),flush=True)
    ck=torch.load(out/'checkpoint_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state'])
    test=evaluate(model,eval_loader('test'),device,classes,save=out/'test_predictions.npz')
    result=dict(identity=identity_info,best_epoch=ck['epoch'],best_validation_axon_macro_f1=ck['best_validation_axon_macro_f1'],
                window_test_metrics=test['window'],test_metrics=test['cell'],classes=classes,
                head_nodes=model.cls_head.nodes,level_classes=h.level_classes,shared_database_predictions_written=False)
    atomic_json(out/'complete.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5))
    p.add_argument('--condition',choices=['axon_only','axon_dendrite'],required=True);p.add_argument('--workers',type=int,default=15)
    p.add_argument('--epochs',type=int,default=30);a=p.parse_args();train(a.fold,a.condition,a.workers,a.epochs)
