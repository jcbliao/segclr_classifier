"""Eight persistent Slurm workers consume an audited, resumable training queue."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_report_fold_sweep import BASE,read,save,sha

def claim(plan,worker):
    preempt=worker>=4
    with (BASE/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(BASE/'queue.json') if (BASE/'queue.json').exists() else {}
        # Resume the task held by this worker before claiming another one.
        eligible=[j for j in plan['jobs'] if (j['index'] in plan['preemptable_indices'])==preempt]
        packing = read(BASE/'packing_plan.json') if (BASE/'packing_plan.json').exists() else None
        if packing:
            from data.packing_cache import ready
            eligible = [j for j in eligible if ready(packing['task_keys'][str(j['index'])])]
        eligible.sort(key=lambda j:(state.get(str(j['index']),{}).get('worker')!=worker,j['index']))
        for job in eligible:
            if job.get('requires') and not Path(job['requires']).exists():continue
            key=str(job['index']);prior=state.get(key,{})
            report=ROOT/job['report']
            if report.exists():
                payload=read(report)
                if 'window_test_metrics' in payload:
                    state[key]={'status':'complete','worker':prior.get('worker',worker)}
                    continue
            if prior.get('status')=='running' and prior.get('worker')!=worker:continue
            if prior.get('attempts',0)>=2:continue
            state[key]={**prior,'worker':worker,'status':'running','slurm_job':os.environ.get('SLURM_JOB_ID'),
                        'started_at':time.time()}
            save(BASE/'queue.json',state);return job
        save(BASE/'queue.json',state)
    return None

def finish(job,worker,status,code):
    with (BASE/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=read(BASE/'queue.json');key=str(job['index'])
        prior=state[key];assert prior['worker']==worker
        prior.update(status=status,exit_code=code,finished_at=time.time())
        if status=='failed':prior['attempts']=prior.get('attempts',0)+1
        save(BASE/'queue.json',state)

def run(worker):
    plan=read(BASE/'plan.json')
    os.chdir(ROOT)
    import torch
    assert torch.cuda.is_available(),'Refusing CPU training fallback'
    while True:
        job=claim(plan,worker)
        if job is None:
            pending=[j for j in plan['jobs'] if (j['index'] in plan['preemptable_indices'])==(worker>=4)
                     and j.get('requires') and not Path(j['requires']).exists()]
            if (BASE/'packing_plan.json').exists():
                from data.packing_cache import ready
                packing=read(BASE/'packing_plan.json')
                pending += [j for j in plan['jobs'] if (j['index'] in plan['preemptable_indices'])==(worker>=4)
                            and not ready(packing['task_keys'][str(j['index'])])]
            if pending:
                print('Waiting for CPU packing/density cache preparation',flush=True);time.sleep(30);continue
            return
        config=ROOT/job['config'];a=read(config)
        assert sha(config)==job['config_sha256'] and sha(a['manifest'])==job['manifest_sha256']
        assert sha(ROOT/job['template'])==job['template_sha256']
        env=os.environ.copy();env['SEGCLR_PREDICTION_CACHE']=f'/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/report_folds_20261012/predictions/task{job["index"]}'
        log=ROOT/'logs'/f'report_fold_task{job["index"]:03d}.log'
        print(f'Worker {worker}: task {job["index"]} {job["category"]}, fold {job["fold"]}',flush=True)
        with log.open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u','scripts/train_gnn.py','--config',str(config)],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            def stop(signum,frame):
                # Leave queue ownership intact: the requeued worker resumes it.
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                raise SystemExit(99)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
            code=child.wait()
        complete=code==0 and (ROOT/job['report']).exists()
        finish(job,worker,'complete' if complete else 'failed',code)
        print(f'Task {job["index"]}: {"complete" if complete else "FAILED"} ({code})',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('worker',type=int);run(ap.parse_args().worker)
