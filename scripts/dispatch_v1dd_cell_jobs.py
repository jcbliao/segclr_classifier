"""Submit one GPU job per ready cell and finish only after all commits exist."""
import argparse
import fcntl
from v1dd_file_lock import acquire_lock
import json
import os
import subprocess
import time
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
BASE=Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')

def run(command):
    return subprocess.run(command,check=True,text=True,capture_output=True).stdout

def dispatch_once(base=BASE,repo=REPO):
    roots=[int(r) for r in pd.read_parquet(base/'cohort.parquet').pt_root_id]
    (base/'embedding_commits').mkdir(exist_ok=True)
    with (base/'gpu_dispatch.lock').open('a') as lock:
        acquire_lock(lock)
        path=base/'gpu_cell_jobs.json'
        jobs=json.loads(path.read_text()) if path.exists() else {}
        candidates=[r for r in roots if not (base/'embedding_commits'/f'{r}.json').exists()
                    and ((base/'inputs_v1'/str(r)/'ready.json').exists() or (base/'embeddings'/f'{r}.npz').exists())]
        if candidates:
            # Recover successful submissions whose ledger save was interrupted.
            active={}
            for line in run(['squeue','-h','-u',os.environ.get('USER','jcbliao'),'-o','%i|%80j']).splitlines():
                jid,name=line.strip().split('|',1);active[name.strip()]=jid
            recorded=[j['job_id'] for j in jobs.values()]
            states={}
            if recorded:
                for line in run(['sacct','-X','-n','-P','-j',','.join(recorded),'-o','JobIDRaw,State']).splitlines():
                    fields=line.split('|')
                    if len(fields)>=2: states[fields[0]]=fields[1].split()[0].rstrip('+')
            for rid in candidates:
                key=str(rid);name=f'v1dd_cell_{rid}'
                entry=jobs.get(key)
                if name in active:
                    jobs[key]=dict(job_id=active[name],attempts=entry['attempts'] if entry else 1)
                    continue
                if entry:
                    state=states.get(entry['job_id'],'UNKNOWN')
                    if state in {'UNKNOWN','PENDING','RUNNING','COMPLETING','REQUEUED','REQUEUE_HOLD','SUSPENDED','CONFIGURING'}: continue
                    if entry['attempts']>=3:
                        raise RuntimeError(f'Cell {rid}: GPU job {entry["job_id"]} ended {state}, three attempts exhausted')
                command=['sbatch','--parsable','--partition=mit_normal_gpu','--account=mit_amf_advanced_gpu',
                    '--qos=mit_amf_advanced_gpu','--cpus-per-task=4','--time=00:30:00',f'--job-name={name}',
                    f'--output={repo}/logs/v1dd_cell_{rid}_%j.out',f'--error={repo}/logs/v1dd_cell_{rid}_%j.err',
                    str(repo/'scripts/sbatch/infer_v1dd_candidates.sh'),'--input-root',key,'--max-seconds','1500']
                jid=run(command).strip().split(';')[0]
                jobs[key]=dict(job_id=jid,attempts=entry['attempts']+1 if entry else 1)
                # Save after each submission, not after the complete dispatch cycle.
                temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(jobs,indent=2)+'\n');temp.replace(path)
                print(f'CELL GPU SUBMITTED root={rid} job={jid}',flush=True)
            temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(jobs,indent=2)+'\n');temp.replace(path)
    committed=sum((base/'embedding_commits'/f'{r}.json').exists() for r in roots)
    print(f'CELL JOB PROGRESS committed={committed}/{len(roots)} submitted={len(jobs)}',flush=True)
    return committed==len(roots)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');ap.add_argument('--max-seconds',type=float,default=42000)
    args=ap.parse_args();deadline=time.monotonic()+args.max_seconds
    while True:
        try:
            if dispatch_once() or args.once: break
        except subprocess.CalledProcessError as exc:
            # Scheduler queries/submissions occasionally fail transiently.
            # Recover submitted jobs by name on the next cycle rather than
            # stopping the dispatcher and invalidating downstream dependencies.
            print(f'SCHEDULER RETRY command={exc.cmd[0]} status={exc.returncode}: {(exc.stderr or str(exc)).strip()}',flush=True)
            if args.once:raise
        if time.monotonic()>deadline: raise SystemExit(75)
        time.sleep(20)
