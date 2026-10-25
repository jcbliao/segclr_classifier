"""Measure compilation startup and steady training with unchanged model/data."""
import sys,json,time,copy,itertools,dataclasses,os,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/jcbliao_segclr_inductor')
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/jcbliao_segclr_triton')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','8')
import torch
from data.packed_presynaptic import PackedPresynapticWindowDataset,ResidentWindowLoader
from data.mixed_cell_batch_sampler import MixedCellBatchSampler
from gnn.model import WindowClassifier
from gnn.cuda_prefetch import CUDABatchPrefetcher
from gnn.training_performance import model_training_loss
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010';OUT=BASE/'performance'

def main():
 torch.set_num_threads(1);plan=json.loads((BASE/'plan.json').read_text());j=next(j for j in plan['jobs'] if j['family']=='CAVE');a=json.loads((ROOT/j['config']).read_text());manifest=json.loads(Path(a['manifest']).read_text())
 train=[r for r,v in manifest['cells'].items() if v['split']=='train'][:128];manifest['cells']={r:manifest['cells'][r] for r in train}
 ds=PackedPresynapticWindowDataset(manifest,'train','cave',database=a['presynaptic_database'],cell_cache_size=16);ds.prepare_packed();assert ds.prepare_device(torch.device('cuda'))
 sampler=MixedCellBatchSampler(ds,cells_per_batch=16,attention_budget=a['attention_budget'],max_windows=4096,shuffle=True,seed=0,balance_classes=True)
 batches=list(itertools.islice(iter(sampler),136));results=[]
 for arch in ['mean','pointwise_mlp','graph_transformer']:
  job=next(j for j in plan['jobs'] if j['family']=='CAVE' and j['architecture']==arch);cfg=torch.load((ROOT/job['template']).with_suffix('')/'checkpoint_best.pt',map_location='cpu',weights_only=False)['config']
  for mode in ['eager','compiled']:
   torch.manual_seed(0);m=WindowClassifier(cfg,ds.hierarchy).cuda().train();opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-5,fused=True);scaler=torch.amp.GradScaler('cuda');fn=(torch.compile(model_training_loss,dynamic=True,mode=os.environ.get('SEGCLR_COMPILE_MODE','reduce-overhead')) if mode=='compiled' else model_training_loss)
   startup=time.perf_counter();times=[]
   for epoch in range(5 if os.environ.get("SEGCLR_VARY_PLANS") else 3):
    if os.environ.get("SEGCLR_VARY_PLANS"):
     sampler.set_epoch(epoch);batches=list(itertools.islice(iter(sampler),136))
    torch.cuda.synchronize();begin=time.perf_counter();count=0
    for b,shape,routes in CUDABatchPrefetcher(ResidentWindowLoader(ds,batches),torch.device('cuda'),m.cls_head):
     if mode=='compiled':torch.compiler.cudagraph_mark_step_begin()
     opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.float16):
      loss=fn(m,b.x,b.edge_index,b.batch,getattr(b,'pos_enc',None),getattr(b,'rel_pos',None),None,b.has_segclr,None,shape,b.y_levels,routes)
     if mode=='compiled' and epoch==0 and count==0:
      reference=copy.deepcopy(m);reference.zero_grad(set_to_none=True)
      with torch.autocast('cuda',dtype=torch.float16):
       expected=model_training_loss(reference,b.x,b.edge_index,b.batch,getattr(b,'pos_enc',None),getattr(b,'rel_pos',None),None,b.has_segclr,None,shape,b.y_levels,routes)
      expected.backward();loss.backward()
      torch.testing.assert_close(loss,expected,rtol=.005,atol=.0005)
      for left,right in zip(m.parameters(),reference.parameters()):
       assert (left.grad is None)==(right.grad is None)
       if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,rtol=.05,atol=.0005)
      print('COMPILED FP16 LOSS/GRADIENT PARITY PASSED',arch,flush=True)
      del reference;opt.zero_grad(set_to_none=True)
      torch.compiler.cudagraph_mark_step_begin()
      with torch.autocast('cuda',dtype=torch.float16):
       loss=fn(m,b.x,b.edge_index,b.batch,getattr(b,'pos_enc',None),getattr(b,'rel_pos',None),None,b.has_segclr,None,shape,b.y_levels,routes)
     scaler.scale(loss).backward();scaler.step(opt);scaler.update();count+=1
    torch.cuda.synchronize();times.append(time.perf_counter()-begin);assert torch.isfinite(loss)
    print(json.dumps(dict(architecture=arch,mode=mode,epoch=epoch,seconds=times[-1],batches=count)),flush=True)
   busy=None
   if os.environ.get('SEGCLR_VARY_PLANS'):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     for b,shape,routes in CUDABatchPrefetcher(ResidentWindowLoader(ds,batches[:40]),torch.device('cuda'),m.cls_head):
      if mode=='compiled':torch.compiler.cudagraph_mark_step_begin()
      opt.zero_grad(set_to_none=True)
      with torch.autocast('cuda',dtype=torch.float16):
       loss=fn(m,b.x,b.edge_index,b.batch,getattr(b,'pos_enc',None),getattr(b,'rel_pos',None),None,b.has_segclr,None,shape,b.y_levels,routes)
      scaler.scale(loss).backward();scaler.step(opt);scaler.update()
     torch.cuda.synchronize()
    events=sorted((e.time_range.start,e.time_range.end) for e in prof.events() if 'CUDA' in str(e.device_type))
    if events:
     merged=[]
     for lo,hi in events:
      if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
      else:merged.append([lo,hi])
     busy=sum(hi-lo for lo,hi in merged)/(merged[-1][1]-merged[0][0])
    print('GPU KERNEL BUSY FRACTION',arch,mode,busy,flush=True)
   results.append(dict(architecture=arch,mode=mode,epoch_seconds=times,total_seconds=time.perf_counter()-startup,kernel_busy_fraction=busy))
   (OUT/('compiled_dynamic_profile.json' if os.environ.get('SEGCLR_VARY_PLANS') else 'compiled_training_profile.json')).write_text(json.dumps(results,indent=2))
   del fn,opt,m;gc.collect();torch.cuda.empty_cache();torch._dynamo.reset()
 print('COMPILED BENCHMARK COMPLETE',flush=True)
if __name__=='__main__':main()
