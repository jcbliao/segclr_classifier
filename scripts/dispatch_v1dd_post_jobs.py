"""Dispatch GPU jobs only for completed shared-CPU post-site shards."""
import json,os,subprocess,time
from pathlib import Path
from prepare_v1dd_post_batch_queue import BASE
from v1dd_file_lock import acquire_lock
REPO=Path(__file__).resolve().parents[1]
next_recovery=0
def run(command):return subprocess.check_output(command,text=True)
def recover_preparation():
 global next_recovery
 if time.monotonic()<next_recovery:return
 next_recovery=time.monotonic()+60
 config=BASE/'preparation_jobs.json'
 if not config.exists():return
 cells=json.loads((BASE/'batch_queue.json').read_text())['cells']
 if not any(c['status']=='OPEN' for c in cells.values()):return
 arrays=json.loads(config.read_text())['job_ids']
 records=run(['sacct','-X','-n','-P','-j',','.join(arrays),'-o','JobIDRaw,State'])
 for line in records.splitlines():
  fields=line.split('|')
  if len(fields)<2:continue
  job,state=fields[0],fields[1].split()[0]
  if state in ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']:
   run(['scontrol','requeue',job])
   print('POST WORKER RECOVERED',job,state,flush=True)
def once():
 recover_preparation()
 m=json.loads((BASE/'manifest.json').read_text());n=m['n_shards']
 with (BASE/'post_dispatch.lock').open('a') as lock:
  acquire_lock(lock);p=BASE/'post_gpu_jobs.json';jobs=json.loads(p.read_text()) if p.exists() else {}
  active={}
  for line in run(['squeue','-h','-u',os.environ.get('USER','jcbliao'),'-o','%i|%80j']).splitlines():
   job,name=line.split('|',1);active[name.strip()]=job.strip()
  for part in range(n):
   if (BASE/'embedding_commits'/f'{part}.json').exists():continue
   if not (BASE/'inputs_v1'/str(part)/'ready.json').exists() and not (BASE/'embeddings'/f'part-{part:05d}.parquet').exists():continue
   name=f'v1dd_post_part_{part}';key=str(part);entry=jobs.get(key)
   if name in active:continue
   if entry:
    states=run(['sacct','-X','-n','-P','-j',entry['job_id'],'-o','State']).splitlines()
    state=states[0].split('|')[0].split()[0] if states else 'UNKNOWN'
    if state in ['UNKNOWN','RUNNING','PENDING','COMPLETING','REQUEUED','CONFIGURING']:continue
    if entry['attempts']>=3:raise RuntimeError(f'Post shard {part}: three GPU attempts failed')
   preempt=part%2==0
   command=['sbatch','--parsable','--partition='+('mit_preemptable' if preempt else 'mit_normal_gpu'),'--account='+('mit_general' if preempt else 'mit_amf_advanced_gpu'),'--qos='+('normal' if preempt else 'mit_amf_advanced_gpu'),'--job-name='+name,'--output='+str(REPO/f'logs/v1dd_post_part_{part}_%j.out'),'--error='+str(REPO/f'logs/v1dd_post_part_{part}_%j.err'),str(REPO/'scripts/sbatch/infer_v1dd_postsynaptic_sites.sh'),'--part',str(part)]
   job=run(command).strip().split(';')[0];jobs[key]=dict(job_id=job,attempts=entry['attempts']+1 if entry else 1)
   t=p.with_suffix('.tmp.json');t.write_text(json.dumps(jobs,indent=2));t.replace(p)
   print('POST GPU SUBMITTED',part,job,flush=True)
 done=sum((BASE/'embedding_commits'/f'{i}.json').exists() for i in range(n));print('POST PROGRESS',done,'/',n,flush=True);return done==n
if __name__=='__main__':
 deadline=time.monotonic()+42000
 while True:
  try:
   if once():break
  except subprocess.CalledProcessError as exc:print('SCHEDULER RETRY',exc.returncode,flush=True)
  if time.monotonic()>deadline:raise SystemExit(75)
  time.sleep(20)
