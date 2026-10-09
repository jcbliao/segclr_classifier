import time,json,re,subprocess
from pathlib import Path
pattern=re.compile(r'INPUT READY root=(\d+) batch=(\d+) nodes=(\d+)')
def batches():
 result={}
 for p in Path('logs').glob('v1dd_batch_*.out'):
  for m in pattern.finditer(p.read_text()):result[m[1]+'/'+m[2]]=int(m[3])
 return result
def workers():
 out=subprocess.check_output(['squeue','-r','-j','24550084,24550197,24551588,24551715','-h','-O','Partition:30,State:20,NumTasks:10'],text=True)
 counts={}
 for line in out.splitlines():
  part,state,n=line.split()
  if state=='RUNNING':counts[part]=counts.get(part,0)+int(n)
 return counts
start=batches();t0=time.time();samples=[];reports=[];previous=start;previous_t=t0
for i in range(19):
 time.sleep(max(0,t0+i*10-time.time()));t=time.time();counts=workers();samples.append(dict(time=t,workers=sum(counts.values()),partitions=counts))
 if i and i%6==0:
  current=batches();end=time.time();new=sum(v for k,v in current.items() if k not in previous)
  recent=[s for s in samples if s['time']>=previous_t-1]
  worker_seconds=sum((b['time']-a['time'])*(a['workers']+b['workers'])/2 for a,b in zip(recent,recent[1:]))
  report=dict(elapsed_seconds=round(end-t0,1),inputs=new,inputs_per_minute=round(new*60/(end-previous_t),1),average_running_workers=round(worker_seconds/(recent[-1]['time']-recent[0]['time']),1),inputs_per_worker_minute=round(new*60/worker_seconds,1),running_now=counts)
  reports.append(report);print(json.dumps(report),flush=True);previous=current;previous_t=t
  Path('analysis/v1dd_cell_types/input_profile/worker_normalized_rate.json').write_text(json.dumps(dict(samples=samples,reports=reports),indent=2))
