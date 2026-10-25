"""Benchmark actual sweep architectures and heads against packed/resident input."""
import sys,json,time,itertools,copy,dataclasses,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.batched_presynaptic import BatchedPresynapticWindowDataset,collate_presynaptic
from data.packed_presynaptic import PackedPresynapticWindowDataset,ResidentWindowLoader
from data.mixed_cell_batch_sampler import MixedCellBatchSampler
from gnn.model import WindowClassifier
from gnn.training_performance import prepare_routes,move_routes,routed_loss
from gnn.cuda_prefetch import CUDABatchPrefetcher
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010'
OUT=BASE/'performance'

def trial(ds,batches,cfg,mode,fused=False):
    torch.manual_seed(0);model=WindowClassifier(cfg,ds.hierarchy).cuda().train()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5,fused=fused);scaler=torch.amp.GradScaler('cuda')
    if mode=='baseline':loader=DataLoader(ds,batch_sampler=batches,num_workers=15,persistent_workers=True,pin_memory=True,collate_fn=collate_presynaptic)
    else:
        loader=ResidentWindowLoader(ds,batches) if mode=='resident' else DataLoader(ds,batch_sampler=batches,num_workers=0,collate_fn=collate_presynaptic)
        loader=CUDABatchPrefetcher(loader,torch.device('cuda'),model.cls_head)
    wait=0.;events=[];windows=0;steps=0;begin=None
    for i,value in enumerate(loader):
        if i==8:torch.cuda.synchronize();begin=time.perf_counter();previous=begin
        if mode=='baseline':
            b=value;shape=(b.num_graphs,int((b.ptr[1:]-b.ptr[:-1]).max()));routes=move_routes(prepare_routes(model.cls_head,b.y_levels),torch.device('cuda'));b=b.to('cuda',non_blocking=True)
        else:b,shape,routes=value
        if i>=8:wait+=time.perf_counter()-previous
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            h=model(b.x,b.edge_index,b.batch,pos_enc=getattr(b,'pos_enc',None),rel_pos=getattr(b,'rel_pos',None),batch_shape=shape)
            loss=routed_loss(model.cls_head,h,b.y_levels,routes)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();end.record()
        if i>=8:events.append((start,end));windows+=shape[0];steps+=1;previous=time.perf_counter()
    torch.cuda.synchronize();elapsed=time.perf_counter()-begin
    assert torch.isfinite(loss)
    result=dict(mode=mode,fused_adam=fused,seconds=elapsed,host_wait_seconds=wait,windows=windows,batches=steps,step_cuda_seconds=sum(a.elapsed_time(b) for a,b in events)/1000)
    if mode=='baseline':loader._iterator._shutdown_workers()
    del loader,opt,model;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    assert torch.cuda.is_available();torch.set_num_threads(1)
    plan=json.loads((BASE/'plan.json').read_text());results=[]
    for family in ['CAVE','TEASAR']:
        job=next(j for j in plan['jobs'] if j['family']==family);a=json.loads((ROOT/job['config']).read_text());manifest=json.loads(Path(a['manifest']).read_text())
        ds=PackedPresynapticWindowDataset(manifest,'train','cave' if family=='CAVE' else 'new',database=a['presynaptic_database'],memmap_root=a.get('presynaptic_memmap_root'),cell_cache_size=16)
        sampler=MixedCellBatchSampler(ds,cells_per_batch=16,attention_budget=a['attention_budget'],max_windows=4096,shuffle=True,seed=0,balance_classes=True)
        batches=list(itertools.islice(iter(sampler),136))
        baseline=copy.copy(ds);baseline.__class__=BatchedPresynapticWindowDataset
        configs={}
        for arch in ['mean','pointwise_mlp','graph_transformer']:
            j=next(j for j in plan['jobs'] if j['family']==family and j['architecture']==arch)
            checkpoint=(ROOT/j['template']).with_suffix('')/'checkpoint_best.pt'
            configs[arch]=torch.load(checkpoint,map_location='cpu',weights_only=False)['config']
            print('ACTUAL CONFIG',family,arch,configs[arch],flush=True)
            baseline.geometry_required=arch=='graph_transformer'
            row=dict(family=family,architecture=arch,**trial(baseline,batches,configs[arch],'baseline'));results.append(row);print(json.dumps(row),flush=True)
        ds.prepare_packed(workers=8)
        assert ds.prepare_device(torch.device('cuda'))
        # Full tensor parity on real roots, repeated draws and mixed-root batches.
        for indices in batches[:3]:
            for fast in [ds.__getitems__(indices),ds.get_batch(indices,device=True).cpu()]:
                reference=baseline.__getitems__(indices)
                # Baseline may omit geometry for mean; core features remain exact.
                for name in reference.keys():
                    torch.testing.assert_close(reference[name],fast[name],rtol=0,atol=0)
        for arch in ['mean','pointwise_mlp','graph_transformer']:
            for mode,fused in [('packed',False),('resident',False),('resident',True)]:
                row=dict(family=family,architecture=arch,packing_seconds=ds.packing_seconds,packing_bytes=ds.packing_bytes,**trial(ds,batches,configs[arch],mode,fused));results.append(row);print(json.dumps(row),flush=True)
                (OUT/'resident_loading_profile.json').write_text(json.dumps(results,indent=2))
        del ds,baseline;gc.collect();torch.cuda.empty_cache()
    from scripts.validate_fp16_training import main as validate_fp16
    validate_fp16()
    print('ALL RESIDENT LOADING BENCHMARKS PASSED',flush=True)
if __name__=='__main__':main()
