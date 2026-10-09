"""Measure new durable input batches and committed cells over a real interval."""
import argparse,json,re,subprocess,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--minutes',type=int,default=5);ap.add_argument('--jobs',default='24545731,24545737');ap.add_argument('--output',required=True);args=ap.parse_args()
BASE=Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196');ROOT=Path(__file__).resolve().parents[3]
pattern=re.compile(r'INPUT READY root=(\d+) batch=(\d+) nodes=(\d+)')
def snapshot():
 batches={}
 for p in (ROOT/'logs').glob('v1dd_batch_*.out'):
  for m in pattern.finditer(p.read_text()):batches[m[1]+'/'+m[2]]=int(m[3])
 commits={p.stem for p in (BASE/'embedding_commits').glob('*.json')}
 counts={};jobs={}
 text=subprocess.check_output(['squeue','-r','-j',args.jobs,'-h','-O','Partition:30,State:15,NumTasks:8'],text=True)
 for line in text.splitlines():
  partition,state,ntasks=line.split();label=partition+' '+state
  counts[label]=counts.get(label,0)+int(ntasks);jobs[label]=jobs.get(label,0)+1
 return dict(time=time.time(),batches=batches,commits=sorted(commits),workers=counts,jobs=jobs)
start=snapshot();previous=start;reports=[];out=Path(args.output)
print(json.dumps(dict(start_time=start['time'],prepared_inputs=sum(start['batches'].values()),committed_cells=len(start['commits']),workers=start['workers'])),flush=True)
for i in range(args.minutes):
 time.sleep(max(0,start['time']+(i+1)*60-time.time()));now=snapshot()
 new=set(now['batches'])-set(start['batches']);nodes=sum(now['batches'][k] for k in new)
 recent=set(now['batches'])-set(previous['batches']);recent_nodes=sum(now['batches'][k] for k in recent)
 report=dict(elapsed_seconds=round(now['time']-start['time'],1),new_inputs=nodes,new_batches=len(new),inputs_per_minute=round(nodes*60/(now['time']-start['time']),1),last_interval_inputs_per_minute=round(recent_nodes*60/(now['time']-previous['time']),1),new_cells=len(set(now['commits'])-set(start['commits'])),workers=now['workers'],jobs=now['jobs'])
 reports.append(report);out.write_text(json.dumps(dict(start=start,measurements=reports),indent=2));print(json.dumps(report),flush=True);previous=now
