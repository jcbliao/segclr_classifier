import copy,json
import numpy as np
import pytest
from scripts.build_report_fold_sweep import assignments,folded
from scripts.train_gnn import selection_domain_mask,build_training_parser

def test_historical_fold_zero_preserved_and_every_cell_held_once():
    source={'cells':{str(i):{'cell_type':'A' if i%2 else 'B','split':'test' if i%5==0 else 'train'} for i in range(100)}}
    original=copy.deepcopy(source);mapping=assignments(source)
    assert mapping==assignments(source)
    manifests=[folded(source,f,mapping) for f in range(5)]
    assert source==original
    assert {r for r,i in manifests[0]['cells'].items() if i['split']=='test'}=={r for r,i in source['cells'].items() if i['split']=='test'}
    for root in source['cells']:assert sum(m['cells'][root]['split']=='test' for m in manifests)==1
    for label in ['A','B']:
        counts=[sum(i['cell_type']==label and i['split']=='test' for i in m['cells'].values()) for m in manifests[1:]]
        assert max(counts)-min(counts)<=1

def test_domain_selection_matches_repeated_actual_roots():
    manifest={'cells':{'1':{'source_dataset':'microns'},'2':{'source_dataset':'v1dd'}}}
    np.testing.assert_array_equal(selection_domain_mask(manifest,np.array([2,1,1,2]),'microns'),[False,True,True,False])
    with pytest.raises(ValueError):selection_domain_mask(manifest,np.array([2,2]),'microns')
    assert build_training_parser().parse_args(['--selection-domain','microns']).selection_domain=='microns'

def test_queue_unique_claims_and_resumes(tmp_path,monkeypatch):
    import scripts.run_report_fold_worker as worker
    monkeypatch.setattr(worker,'BASE',tmp_path)
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    plan={'jobs':[{'index':i,'report':f'report{i}.json'} for i in range(12)],'preemptable_indices':[0,4,8]}
    a=worker.claim(plan,0);b=worker.claim(plan,1);c=worker.claim(plan,4)
    assert len({a['index'],b['index'],c['index']})==3
    assert c['index'] in plan['preemptable_indices']
    assert worker.claim(plan,0)==a
    worker.finish(a,0,'failed',1)
    assert worker.claim(plan,0)==a
    (tmp_path/a['report']).write_text(json.dumps({'window_test_metrics':{}}))
    assert worker.claim(plan,0)['index']!=a['index']

def test_five_fold_interval_and_incomplete_group_exclusion():
    from analysis.report_folds_20261012.all_folds import mean_ci,complete_records
    import pandas as pd
    mean,low,high=mean_ci([1,2,3,4,5])
    assert mean==3
    assert high-mean==pytest.approx(1.963243,abs=1e-6)
    assert mean-low==pytest.approx(high-mean)
    with pytest.raises(ValueError):mean_ci([1,2,3,4])
    cache={'status':pd.DataFrame([{'group':0,'complete':False},{'group':1,'complete':True}]),
           'records':[{'group':0},{'group':1}]}
    assert complete_records(cache)==[{'group':1}]
