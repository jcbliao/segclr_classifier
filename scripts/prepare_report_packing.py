"""Build one reusable dataset cache on a CPU node; never initialize CUDA."""
import argparse,json,os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_gnn import build_training_parser,build_training_datasets
from scripts.training_config import parse_training_args
from data.packing_cache import identity,prepare_cached,ready
from data.packed_windows import PackedWholeCellWindows
import torch

def release_workers(item):
    for job in item.get('release_jobs',[]):
        result=subprocess.run(['scontrol','release',job],capture_output=True,text=True)
        print(f'Release {job}: {result.returncode} {result.stderr.strip()}',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('index',type=int);args=p.parse_args()
    base=ROOT/'analysis/report_folds_20261012'
    plan=json.loads((base/'packing_plan.json').read_text())
    item=plan['jobs'][args.index]
    config=parse_training_args(build_training_parser(),['--config',str(ROOT/item['config'])])
    manifest=json.loads(Path(config.manifest).read_text())
    key=identity(config,manifest)
    assert key==item['key'],'Packing source inputs/code changed after submission'
    if ready(key):
        print(f'Already cached {key}',flush=True)
        release_workers(item)
        return
    torch.set_num_threads(1)
    train,test,_=build_training_datasets(config)
    if config.dataset=='all_windows':train,test=PackedWholeCellWindows(train),PackedWholeCellWindows(test)
    workers=max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','16'))-1)
    for split,dataset in [('train',train),('test',test)]:
        # Disk/CPU cache capacity is separate from the GPU residency budget.
        # Keep existing cache identities: only the build allowance changes.
        if os.environ.get('SEGCLR_CPU_PACKING_LIMIT_GIB'):
            original_prepare = dataset.prepare_packed
            cpu_limit = int(os.environ['SEGCLR_CPU_PACKING_LIMIT_GIB']) * 2**30
            def prepare_with_cpu_budget(_original=original_prepare, **kwargs):
                kwargs['max_bytes'] = cpu_limit
                return _original(**kwargs)
            dataset.prepare_packed = prepare_with_cpu_budget
        prepare_cached(dataset,key,split,workers=workers,build=True)
    print(f'CPU packing complete {key}',flush=True)
    release_workers(item)

if __name__=='__main__':main()
