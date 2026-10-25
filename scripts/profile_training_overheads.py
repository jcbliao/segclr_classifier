"""Measure actual GPU training stages on identical fixed presynaptic batches."""
import sys,json,time,copy,itertools,statistics,os,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.batched_presynaptic import BatchedPresynapticWindowDataset,collate_presynaptic
from data.mixed_cell_batch_sampler import MixedCellBatchSampler
from data.dataset_lcpn import load_hierarchy
from gnn.model import ModelConfig,WindowClassifier
from gnn.training_performance import prepare_routes,move_routes,routed_loss,static_top_down
from gnn.metrics import majority_vote_by_group
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010';OUT=BASE/'performance'

def reference_sampler():
    spec=importlib.util.spec_from_file_location('reference_dataset_presynaptic',OUT/'reference_dataset_presynaptic.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace={};source=(OUT/'reference_mixed_cell_batch_sampler.py').read_text().replace('from data.dataset_presynaptic import AttentionBudgetBatchSampler','')
    namespace['AttentionBudgetBatchSampler']=module.AttentionBudgetBatchSampler;exec(compile(source,'reference_mixed_cell_batch_sampler.py','exec'),namespace)
    return namespace['MixedCellBatchSampler']

def legacy_vote(roots,true,pred):
    groups=np.unique(roots);yt=np.empty(len(groups),dtype=true.dtype);yp=np.empty(len(groups),dtype=pred.dtype)
    for i,g in enumerate(groups):
        mask=roots==g;yt[i]=true[mask][0];yp[i]=np.bincount(pred[mask]).argmax()
    return yt,yp

def forward(model,batch,optimized):
    kw=dict(pos_enc=getattr(batch,'pos_enc',None),rel_pos=getattr(batch,'rel_pos',None))
    if optimized and getattr(model,'supports_batch_shape',False):kw['batch_shape']=(batch.num_graphs,int(batch._profile_max_nodes))
    return model(batch.x,batch.edge_index,batch.batch,**kw)

@torch.inference_mode()
def evaluation_trial(model,loader,device,optimized):
    predictions=[];labels=[];roots=[]
    torch.cuda.synchronize();begin=time.perf_counter()
    if optimized:
        from scripts.train_gnn import evaluate
        labels, predictions, roots = evaluate(model, iter(loader), device, amp=True)
        torch.cuda.synchronize()
        return time.perf_counter()-begin, predictions, labels, roots
    for b in iter(loader):
        if optimized:
            labels.append(b.y_levels[:,-1].numpy());roots.append(b.root_id.numpy())
        b._profile_max_nodes=int((b.ptr[1:]-b.ptr[:-1]).max());b=b.to(device,non_blocking=optimized)
        with torch.autocast('cuda',dtype=torch.float16):
            h=forward(model,b,optimized)
            pred=(static_top_down(model.cls_head,h) if optimized else model.cls_head.predict_top_down(h))[:,-1]
        predictions.append(pred if optimized else pred.cpu())
        if not optimized:
            labels.append(b.y_levels[:,-1].cpu().numpy());roots.append(b.root_id.cpu().numpy())
    result=torch.cat(predictions).cpu().numpy();torch.cuda.synchronize()
    return time.perf_counter()-begin,result,np.concatenate(labels),np.concatenate(roots)

def run(stage):
    optimized=stage=='optimized';torch.set_num_threads(1);assert torch.cuda.is_available();device=torch.device('cuda')
    plan=json.loads((BASE/'plan.json').read_text());results=[]
    for family in ['CAVE','TEASAR']:
        j=next(j for j in plan['jobs'] if j['family']==family);a=json.loads((ROOT/j['config']).read_text());manifest=json.loads(Path(a['manifest']).read_text())
        train=[r for r,v in manifest['cells'].items() if v['split']=='train'][:64];manifest['cells']={r:manifest['cells'][r] for r in train}
        begin=time.perf_counter();ds=BatchedPresynapticWindowDataset(manifest,'train','cave' if family=='CAVE' else 'new',database=a['presynaptic_database'],memmap_root=a.get('presynaptic_memmap_root'),cell_cache_size=16);startup=time.perf_counter()-begin
        sampler_args=dict(cells_per_batch=16,attention_budget=a['attention_budget'],max_windows=4096,shuffle=True,seed=0,balance_classes=True)
        begin=time.perf_counter();sampler=(MixedCellBatchSampler if optimized else reference_sampler())(ds,**sampler_args);batches=list(itertools.islice(iter(sampler),80));sampling=time.perf_counter()-begin
        if optimized:
            ref=reference_sampler()(ds,**sampler_args);assert list(iter(sampler))==list(iter(ref))
        loader=DataLoader(ds,batch_sampler=batches,num_workers=15,persistent_workers=True,pin_memory=optimized,collate_fn=collate_presynaptic)
        for arch in ['mean','pointwise_mlp','graph_transformer']:
            if optimized:
                if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
                ds.geometry_required=arch=='graph_transformer'
                loader=DataLoader(ds,batch_sampler=batches,num_workers=15,persistent_workers=True,pin_memory=True,collate_fn=collate_presynaptic)
            torch.manual_seed(0);model=WindowClassifier(ModelConfig(architecture=arch),ds.hierarchy).to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5);scaler=torch.amp.GradScaler('cuda')
            copy_events=[];compute_events=[];wait=0.;host_copy=0.;windows=0;loss_sum=torch.zeros((),device=device,dtype=torch.float64);iterator=iter(loader)
            # Warm workers and CUDA kernels before timing.
            for _ in range(8):
                b=next(iterator);routes=prepare_routes(model.cls_head,b.y_levels) if optimized else None
                b._profile_max_nodes=int((b.ptr[1:]-b.ptr[:-1]).max());b=b.to(device,non_blocking=optimized);routes=move_routes(routes,device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    h=forward(model,b,optimized);loss=routed_loss(model.cls_head,h,b.y_levels,routes) if optimized else model.cls_head.compute_loss(h,b.y_levels)
                scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(min(64,len(batches)-8)):
                t=time.perf_counter();b=next(iterator);wait+=time.perf_counter()-t
                routes=prepare_routes(model.cls_head,b.y_levels) if optimized else None
                b._profile_max_nodes=int((b.ptr[1:]-b.ptr[:-1]).max());t=time.perf_counter();cs=torch.cuda.Event(enable_timing=True);ce=torch.cuda.Event(enable_timing=True);cs.record()
                b=b.to(device,non_blocking=optimized);routes=move_routes(routes,device);ce.record();host_copy+=time.perf_counter()-t;copy_events.append((cs,ce))
                bs=torch.cuda.Event(enable_timing=True);be=torch.cuda.Event(enable_timing=True);bs.record();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    h=forward(model,b,optimized);loss=routed_loss(model.cls_head,h,b.y_levels,routes) if optimized else model.cls_head.compute_loss(h,b.y_levels)
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();be.record();compute_events.append((bs,be));windows+=b.num_graphs
                if optimized:loss_sum+=loss.detach().double()*b.num_graphs
                else:float(loss.item())
            torch.cuda.synchronize();wall=time.perf_counter()-begin
            # Identical finite loss, gradients and predictions on real GPU inputs.
            routes=move_routes(prepare_routes(model.cls_head,b.y_levels.cpu()),device)
            model.eval()
            with torch.autocast('cuda',dtype=torch.float16):
                h=forward(model,b,False)
                if optimized:torch.testing.assert_close(h,forward(model,b,True),rtol=0.02,atol=0.032)
                old_loss=model.cls_head.compute_loss(h,b.y_levels);new_loss=routed_loss(model.cls_head,h,b.y_levels,routes)
                old_pred=model.cls_head.predict_top_down(h);new_pred=static_top_down(model.cls_head,h)
            torch.testing.assert_close(old_loss,new_loss,rtol=1e-6,atol=1e-7);assert torch.equal(old_pred,new_pred)
            ck=OUT/f'checkpoint_benchmark_{family}_{arch}.pt';torch.cuda.synchronize();t=time.perf_counter();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),ck);checkpoint=time.perf_counter()-t;ck.unlink()
            # Short profiler capture makes hidden synchronization counts explicit.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(4):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        h=forward(model,b,optimized);loss=routed_loss(model.cls_head,h,b.y_levels,routes) if optimized else model.cls_head.compute_loss(h,b.y_levels)
                    scaler.scale(loss).backward();scaler.step(opt);scaler.update()
                    if not optimized:loss.item()
            sync={e.key:dict(count=e.count,cpu_seconds=e.self_cpu_time_total/1e6) for e in prof.key_averages() if e.key in {'aten::item','aten::nonzero','aten::is_nonzero','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync'}}
            evaluation={}
            if optimized:
                trials={False:[],True:[]};reference=None; disagreements=[]
                for mode in [False,True,True,False]:
                    sec,pred,labels,roots=evaluation_trial(model,loader,device,mode);trials[mode].append(sec)
                    arrays=(pred,labels,roots)
                    if reference is None:reference=arrays
                    else:
                        for x,y in zip(reference[1:],arrays[1:]):assert np.array_equal(x,y)
                        disagreement=float(np.mean(reference[0]!=arrays[0]));disagreements.append(disagreement)
                        assert disagreement < 0.001, disagreement
                evaluation={'baseline_seconds':statistics.median(trials[False]),'optimized_seconds':statistics.median(trials[True]),'prediction_disagreement_fractions':disagreements, 'note':'Independent FP16 scatter reductions can differ near decision boundaries; same-hidden routing checked exactly.'}
            results.append(dict(evaluation=evaluation,family=family,architecture=arch,dataset_startup_seconds=startup,sampling_seconds=sampling,windows=windows,training_wall_seconds=wall,loader_wait_seconds=wait,copy_host_seconds=host_copy,copy_gpu_seconds=sum(s.elapsed_time(e) for s,e in copy_events)/1000,compute_gpu_seconds=sum(s.elapsed_time(e) for s,e in compute_events)/1000,checkpoint_seconds=checkpoint,sync_profile_four_steps=sync,gpu_parity=True))
            print(json.dumps(results[-1]),flush=True)
            (OUT/f'gpu_profile_{stage}_partial.json').write_text(json.dumps(results,indent=2))
        loader._iterator._shutdown_workers();del loader
    rng=np.random.default_rng(0);roots=rng.integers(0,500,450000);true=np.zeros(len(roots),int);pred=rng.integers(0,6,len(roots));vote=majority_vote_by_group if optimized else legacy_vote
    t=time.perf_counter();actual=vote(roots,true,pred);voting=time.perf_counter()-t;expected=legacy_vote(roots,true,pred)
    for x,y in zip(actual,expected):assert np.array_equal(x,y)
    payload=dict(stage=stage,amp_dtype='float16',gpu=torch.cuda.get_device_name(),results=results,voting_450k_seconds=voting)
    (OUT/f'gpu_profile_{stage}.json').write_text(json.dumps(payload,indent=2)+'\n')
    if optimized:
        from scripts.validate_fp16_training import main as validate_fp16
        validate_fp16()
if __name__=='__main__':run(sys.argv[1])
