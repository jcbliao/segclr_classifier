"""Run the four oversized historical TEASAR folds on explicitly requested H200s."""
import argparse,json,os,signal,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_report_fold_worker import BASE,read,save,sha,finish

def main():
    parser=argparse.ArgumentParser();parser.add_argument('index',type=int);args=parser.parse_args()
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory>100*2**30,'H200-class memory required'
    plan=read(BASE/'plan.json');job=next(j for j in plan['jobs'] if j['index']==args.index)
    worker=1000+args.index
    with (BASE/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=read(BASE/'queue.json');prior=state[str(args.index)]
        assert prior['worker']==worker
        prior.update(status='running',execution='h200_running',slurm_job=os.environ.get('SLURM_JOB_ID'),started_at=time.time())
        save(BASE/'queue.json',state)
    config=ROOT/job['config'];a=read(config)
    assert sha(config)==job['config_sha256'] and sha(a['manifest'])==job['manifest_sha256']
    assert sha(ROOT/job['template'])==job['template_sha256']
    env=os.environ.copy();env['SEGCLR_PREDICTION_CACHE']=f'/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/report_folds_20261012/predictions/task{args.index}'
    env['SEGCLR_PACKING_CACHE_ONLY']='1'
    with (ROOT/'logs'/f'report_fold_task{args.index:03}.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,'-u','scripts/train_gnn.py','--config',str(config)],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        def stop(signum,frame):
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            raise SystemExit(99)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
        code=child.wait()
    complete=code==0 and (ROOT/job['report']).exists()
    finish(job,worker,'complete' if complete else 'failed',code)
    if not complete:raise SystemExit(code or 1)

if __name__=='__main__':main()
