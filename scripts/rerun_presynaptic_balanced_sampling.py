"""Replay frozen presynaptic runs using balanced sampling; preserve originals."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PLAN=ROOT/'analysis/presynaptic/balanced_sampling_20261009/plan.json'

def validate(job):
    import hashlib
    import numpy as np
    from data.dataset_presynaptic import PresynapticWindowDataset
    a=json.loads(Path(job['config']).read_text());manifest=Path(a['manifest'])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==job['manifest_sha256']
    assert hashlib.sha256((ROOT/job['source_report']).read_bytes()).hexdigest()==job['source_sha256']
    source=json.loads((ROOT/job['source_report']).read_text())['args']
    allowed={'class_balance','manifest','results_dir','resume','num_workers'}
    assert all(a[k]==v for k,v in source.items() if k not in allowed)
    assert a['class_balance']=='sample'
    variant='cave' if a['dataset']=='presynaptic_cave' else 'new'
    counts={}
    for split in ['train','test']:
        ds=PresynapticWindowDataset(json.loads(manifest.read_text()),split,variant,
                                   database=a['presynaptic_database'],cell_cache_size=1)
        assert ds.classes==job['classes'],(job['index'],ds.classes,job['classes'])
        counts[split]=len(ds)
        if split=='test':assert len(ds)==job['expected_test_windows'],(job['index'],len(ds),job['expected_test_windows'])
    return dict(index=job['index'],architecture=job['architecture'],**counts)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    jobs=json.loads(PLAN.read_text())['jobs']
    if a.preflight:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(validate,jobs))
        (PLAN.parent/'preflight.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results),flush=True)
    else:
        job=jobs[a.index];validate(job)
        os.chdir(ROOT)
        os.execv(sys.executable,[sys.executable,'-u','scripts/train_gnn.py','--config',job['config']])
