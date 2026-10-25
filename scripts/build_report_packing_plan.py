"""Deduplicate immutable input caches across the frozen training queue."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_gnn import build_training_parser
from scripts.training_config import parse_training_args
from data.packing_cache import identity,DEFAULT_ROOT
b=ROOT/'analysis/report_folds_20261012';plan=json.loads((b/'plan.json').read_text());queue=json.loads((b/'queue.json').read_text())
ordered=sorted(plan['jobs'],key=lambda j:(queue.get(str(j['index']),{}).get('status')!='running',j['index']))
jobs={};taskkeys={}
for job in ordered:
    args=parse_training_args(build_training_parser(),['--config',str(ROOT/job['config'])]);manifest=json.loads(Path(args.manifest).read_text());key=identity(args,manifest)
    taskkeys[str(job['index'])]=key
    item=jobs.setdefault(key,dict(key=key,config=job['config'],tasks=[],release_jobs=[]));item['tasks'].append(job['index'])
    state=queue.get(str(job['index']),{})
    if state.get('status')=='running':
        w=state['worker'];item['release_jobs'].append(f'{25641194 if w<4 else 25641195}_{w}')
result=dict(root=str(DEFAULT_ROOT),jobs=list(jobs.values()),task_keys=taskkeys)
(b/'packing_plan.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'{len(jobs)} CPU builders for {len(taskkeys)} training runs',flush=True)
