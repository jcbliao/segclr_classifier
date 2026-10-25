"""Keep partition concurrency independent and avoid duplicate wave submissions."""
import json
import pytest
from scripts import continue_cave_teasar_confidence_sweep as scheduler

@pytest.mark.parametrize('start',[30,90])
def test_waves_chain_preemptable_but_keep_normal_progress_independent(tmp_path,monkeypatch,start):
    monkeypatch.setattr(scheduler,'BASE',tmp_path);monkeypatch.setenv('CONFIDENCE_NEXT_START',str(start))
    (tmp_path/'plan.json').write_text(json.dumps({'jobs':[{}]*120}))
    (tmp_path/'jobs.json').write_text(json.dumps({'waves':[{'first_task':0,'preemptable_training_job':20}],'preemptable_task_indices':[4]+list(range(7,120,4))}))
    calls=[]
    def submit(*args):calls.append(args);return str(100+len(calls)-1)
    monkeypatch.setattr(scheduler,'submit',submit);scheduler.main()
    assert calls[0][0].endswith('%4')
    assert '--dependency=afterany:20' in calls[1]
    assert any(a.endswith('%4') for a in calls[1])
    expected='--dependency=afterany:100:101' if start==90 else '--dependency=afterany:100'
    assert calls[2][0]==expected
    # Calling the continuation twice preserves the existing wave/jobs.
    scheduler.main();assert len(calls)==3
