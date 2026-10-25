"""Full-step CUDA graph benchmark with actual varying epoch sampling plans."""
import sys,json,time,itertools,copy,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from data.packed_presynaptic import PackedPresynapticWindowDataset,ResidentWindowLoader
from data.mixed_cell_batch_sampler import MixedCellBatchSampler
from gnn.model import WindowClassifier
from gnn.cuda_prefetch import CUDABatchPrefetcher
from gnn.cuda_graph_training import CUDAGraphTrainer
from gnn.window_graph_training import WindowGraphTrainer
from gnn.training_performance import model_training_loss
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010';OUT=BASE/'performance'

def main():
 family=sys.argv[1] if len(sys.argv)>1 else 'CAVE';torch.set_num_threads(1);plan=json.loads((BASE/'plan.json').read_text());j=next(j for j in plan['jobs'] if j['family']==family);a=json.loads((ROOT/j['config']).read_text());manifest=json.loads(Path(a['manifest']).read_text());train=[r for r,v in manifest['cells'].items() if v['split']=='train'][:128];manifest['cells']={r:manifest['cells'][r] for r in train}
 ds=PackedPresynapticWindowDataset(manifest,'train','cave' if family=='CAVE' else 'new',database=a['presynaptic_database'],memmap_root=a.get('presynaptic_memmap_root'),cell_cache_size=16);ds.prepare_packed();assert ds.prepare_device(torch.device('cuda'))
 sampler=MixedCellBatchSampler(ds,cells_per_batch=16,attention_budget=a['attention_budget'],max_windows=4096,shuffle=True,seed=0,balance_classes=True);results=[]
 for arch in ['mean','pointwise_mlp','graph_transformer']:
  job=next(j for j in plan['jobs'] if j['family']==family and j['architecture']==arch);cfg=torch.load((ROOT/job['template']).with_suffix('')/'checkpoint_best.pt',map_location='cpu',weights_only=False)['config']
  for mode in ['eager','graph']:
   torch.manual_seed(0);m=WindowClassifier(cfg,ds.hierarchy).cuda().train();opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5,fused=True);scaler=torch.amp.GradScaler('cuda');runner=WindowGraphTrainer(m,opt,scaler,ds) if mode=='graph' else None;times=[]
   for epoch in range(8):
    sampler.set_epoch(epoch);batches=list(itertools.islice(iter(sampler),136));torch.cuda.synchronize();begin=time.perf_counter()
    if runner:runner.loss_sum.zero_()
    if runner:
     for indices in batches:runner.step_indices(indices)
    for b,shape,routes in CUDABatchPrefetcher(ResidentWindowLoader(ds,[] if runner else batches),torch.device('cuda'),m.cls_head,dense=runner is not None):
     if runner:runner.step(b,shape,routes)
     else:
      opt.zero_grad(set_to_none=True)
      with torch.autocast('cuda',dtype=torch.float16):loss=model_training_loss(m,b.x,b.edge_index,b.batch,b.pos_enc,b.rel_pos,None,b.has_segclr,None,shape,b.y_levels,routes)
      scaler.scale(loss).backward();scaler.step(opt);scaler.update()
    torch.cuda.synchronize();times.append(time.perf_counter()-begin)
    if runner:assert torch.isfinite(runner.loss_sum)
    print(json.dumps(dict(architecture=arch,mode=mode,epoch=epoch,seconds=times[-1],graphs=len(runner.cache) if runner else None,hits=runner.hits if runner else None,misses=runner.misses if runner else None)),flush=True)
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    if runner:
     for indices in batches[:40]:runner.step_indices(indices)
    else:
     for b,shape,routes in CUDABatchPrefetcher(ResidentWindowLoader(ds,batches[:40]),torch.device('cuda'),m.cls_head):
      opt.zero_grad(set_to_none=True)
      with torch.autocast('cuda',dtype=torch.float16):loss=model_training_loss(m,b.x,b.edge_index,b.batch,b.pos_enc,b.rel_pos,None,b.has_segclr,None,shape,b.y_levels,routes)
      scaler.scale(loss).backward();scaler.step(opt);scaler.update()
    torch.cuda.synchronize()
   events=sorted((e.time_range.start,e.time_range.end) for e in prof.events() if 'CUDA' in str(e.device_type));merged=[]
   for lo,hi in events:
    if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
    else:merged.append([lo,hi])
   busy=sum(hi-lo for lo,hi in merged)/(merged[-1][1]-merged[0][0]) if merged else None
   print('GPU KERNEL BUSY FRACTION',family,arch,mode,busy,flush=True)
   results.append(dict(family=family,architecture=arch,mode=mode,seconds=times,kernel_busy_fraction=busy));(OUT/f'bucket_graph_profile_{family}.json').write_text(json.dumps(results,indent=2))
   del runner,opt,m;gc.collect();torch.cuda.empty_cache()
 print('FULL STEP GRAPH BENCHMARK COMPLETE',flush=True)
if __name__=='__main__':main()
