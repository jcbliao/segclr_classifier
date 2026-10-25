"""Precompute identical seeded sampler plans using CPU-only worker processes."""
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import numpy as np
from data.dataset_presynaptic import AttentionBudgetBatchSampler
from data.mixed_cell_batch_sampler import MixedCellBatchSampler

@dataclass
class SamplingData:
    index_root_ids: np.ndarray
    index_labels: np.ndarray
    window_sizes: np.ndarray
    def __len__(self):return len(self.index_root_ids)

_worker_sampler=None

def initialize_sampler(data,kwargs,mixed):
    global _worker_sampler
    _worker_sampler=(MixedCellBatchSampler if mixed else AttentionBudgetBatchSampler)(data,**kwargs)

def calculate_plan(epoch):
    _worker_sampler.set_epoch(epoch)
    dtype=np.int32 if len(_worker_sampler.dataset)<2**31 else np.int64
    return epoch,[np.asarray(batch,dtype=dtype) for batch in _worker_sampler]

def precompute_epoch_plans(sampler,start,end,workers=8,parallel_threshold=100000):
    started=time.perf_counter();data=SamplingData(sampler.dataset.index_root_ids,sampler.dataset.index_labels,sampler.dataset.window_sizes)
    keys=('attention_budget','max_windows','shuffle','seed','cell_batch_group','balance_classes','class_balance_power')
    kwargs={key:getattr(sampler,key) for key in keys};mixed=isinstance(sampler,MixedCellBatchSampler)
    if mixed:kwargs['cells_per_batch']=sampler.cells_per_batch
    epochs=list(range(start,end))
    if not epochs:return {}
    if len(data)<parallel_threshold or workers<2:
        # A private sampler keeps the caller's currently selected epoch/plan intact.
        local=(MixedCellBatchSampler if mixed else AttentionBudgetBatchSampler)(data,**kwargs);plans={}
        for epoch in epochs:
            local.set_epoch(epoch);plans[epoch]=[np.asarray(batch,dtype=np.int32 if len(data)<2**31 else np.int64) for batch in local]
    else:
        with ProcessPoolExecutor(max_workers=min(workers,len(epochs)),mp_context=multiprocessing.get_context('spawn'),initializer=initialize_sampler,initargs=(data,kwargs,mixed)) as pool:
            plans=dict(pool.map(calculate_plan,epochs))
    nbytes=sum(batch.nbytes for plan in plans.values() for batch in plan)
    print(f'Prepared {len(plans)} exact epoch plans in {time.perf_counter()-started:.1f}s ({nbytes/2**20:.1f} MiB)',flush=True)
    return plans
