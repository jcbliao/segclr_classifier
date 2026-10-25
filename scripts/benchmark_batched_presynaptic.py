"""Real-cache parity and old/new loading throughput on identical batches."""
import copy,itertools,json,sys,time,statistics,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphLoader
from torch.utils.data import DataLoader
from data.dataset_presynaptic import PresynapticWindowDataset
from data.batched_presynaptic import BatchedPresynapticWindowDataset,collate_presynaptic
from data.mixed_cell_batch_sampler import MixedCellBatchSampler
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010'

def run(family):
    torch.set_num_threads(1)
    plan=json.loads((BASE/'plan.json').read_text());j=next(j for j in plan['jobs'] if j['family']==family)
    a=json.loads((ROOT/j['config']).read_text());manifest=json.loads(Path(a['manifest']).read_text());train=[r for r,v in manifest['cells'].items() if v['split']=='train'][:32]
    manifest['cells']={r:manifest['cells'][r] for r in train}
    ds=BatchedPresynapticWindowDataset(manifest,'train','cave' if family=='CAVE' else 'new',database=a['presynaptic_database'],memmap_root=a.get('presynaptic_memmap_root'),cell_cache_size=32)
    sampler=MixedCellBatchSampler(ds,cells_per_batch=16,attention_budget=a['attention_budget'],max_windows=a['batch_size'],shuffle=True,seed=0,balance_classes=True,class_balance_power=.5)
    batches=list(itertools.islice(iter(sampler),16));assert batches
    parity=[]
    for indices in batches[:3]:
        old=Batch.from_data_list([ds[i] for i in indices]);new=ds.__getitems__(indices)
        assert set(old.keys())==set(new.keys())
        for key in old.keys():torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        parity.append(len(indices))
    times={'old':[],'new':[]}
    for repeat in range(3):
        for kind in (['old','new'] if repeat%2==0 else ['new','old']):
            begin=time.perf_counter()
            for indices in batches:
                value=Batch.from_data_list([ds[i] for i in indices]) if kind=='old' else ds.__getitems__(indices)
            times[kind].append(time.perf_counter()-begin)
    # Exercise multiprocess transport of preassembled batches as well.
    old_ds=copy.copy(ds);old_ds.__class__=PresynapticWindowDataset
    worker_times={}
    for kind in ['old','new']:
        loader=(GraphLoader(old_ds,batch_sampler=batches,num_workers=4,persistent_workers=True) if kind=='old' else DataLoader(ds,batch_sampler=batches,num_workers=4,persistent_workers=True,collate_fn=collate_presynaptic))
        for batch in loader:pass
        begin=time.perf_counter();n=0
        for batch in loader:n+=batch.num_graphs
        worker_times[kind]=time.perf_counter()-begin
        loader._iterator._shutdown_workers();del loader
        assert n==sum(map(len,batches))
    med={k:statistics.median(v) for k,v in times.items()}
    result=dict(family=family,parity_batches=parity,exact_tensor_parity=True,windows=sum(map(len,batches)),batches=len(batches),single_worker_seconds=med,single_worker_speedup=med['old']/med['new'],four_worker_seconds=worker_times,four_worker_speedup=worker_times['old']/worker_times['new'],raw_timings=times)
    (BASE/f'batch_loading_benchmark_{family}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':run(sys.argv[1])
