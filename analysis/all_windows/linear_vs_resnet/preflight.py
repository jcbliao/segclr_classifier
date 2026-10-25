"""Verify the frozen comparison matches historical test support and settings."""
import hashlib,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from data.dataset_lcpn import load_hierarchy,split_cells
BASE=Path(__file__).resolve().parent

def audit():
    plan=json.loads((BASE/'plan.json').read_text());mp=BASE/'manifest.json'
    assert hashlib.sha256(mp.read_bytes()).hexdigest()==plan['manifest_sha256']
    manifest=json.loads(mp.read_text());h=load_hierarchy(manifest);classes=h.level_classes[-1]
    root=Path(json.loads((ROOT/plan['jobs'][0]['config']).read_text())['neighborhood_root'])
    def read(item):
        rid,info=item;p=root/'neighborhoods/n10'/f'{rid}.npz'
        if not p.exists():return dict(root_id=rid,split=info['split'],missing=True)
        assert (ROOT/'data/graph_cache'/f'{rid}.pt').exists()
        with np.load(p) as z:n=len(z['cache_index'])
        with np.load(root/'soma_restricted'/f'{rid}.npz') as z:
            cut=bool(z['cut_applied'][0]);radius=float(z['soma_radius_nm'][0])
            assert not cut or radius==5000
        label=h.level_maps[-1][h.label_paths[info['cell_type']][-1]]
        return dict(root_id=rid,split=info['split'],missing=False,windows=n,label=label,cut_applied=cut)
    items=split_cells(manifest,'train',h)+split_cells(manifest,'test',h)
    with ThreadPoolExecutor(max_workers=16) as pool:rows=list(pool.map(read,items))
    expected=np.zeros(len(classes),dtype=np.int64)
    for r in rows:
        if not r['missing'] and r['split']=='test':expected[r['label']]+=r['windows']
    for job in plan['jobs']:
        p=ROOT/job['source_report'];assert hashlib.sha256(p.read_bytes()).hexdigest()==job['source_sha256']
        report=json.loads(p.read_text());config=json.loads((ROOT/job['config']).read_text())
        assert report['classes']==classes
        assert np.array_equal(np.asarray(report['window_test_metrics']['confusion_matrix']).sum(1),expected)
        allowed={'cls_resnet','num_workers','resume','manifest','results_dir','dataset'}
        assert all(config[k]==v for k,v in report['args'].items() if k not in allowed)
        assert not config['cls_resnet'] and not config['freeze_aggregator'] and config['cls_hidden_dim'] is None
    result=dict(classes=classes,test_class_windows=expected.tolist(),
                splits={s:dict(cells=sum(not r['missing'] and r['split']==s for r in rows),windows=sum(r.get('windows',0) for r in rows if r['split']==s)) for s in ['train','test']},
                uncut_roots=[r['root_id'] for r in rows if not r['missing'] and not r['cut_applied']],
                missing_membership_roots=[r['root_id'] for r in rows if r['missing']],
                roots=rows,matched_historical_test_support=True)
    (BASE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='roots'}),flush=True)

if __name__=='__main__':audit()
