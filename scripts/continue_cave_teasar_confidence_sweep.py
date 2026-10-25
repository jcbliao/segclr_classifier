"""Submit the next wave under the partition's 64-submitted-job limit."""
import json,subprocess,os,datetime,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010'
def submit(*args):
    return subprocess.check_output(['sbatch','--parsable',*args],cwd=ROOT,text=True).strip().split(';')[0]
def main():
    start=int(os.environ.get('CONFIDENCE_NEXT_START','30'));plan=json.loads((BASE/'plan.json').read_text());total=len(plan['jobs'])
    if start>=total:
        os.environ['MORPHO_FOLDS']='0,1,2,3,4'
        # Separate summaries are independent; failures remain explicit in the ledger/log.
        from concurrent.futures import ThreadPoolExecutor
        def summarize(name):return subprocess.run([str(ROOT/'segclr_db/.venv/bin/python'),str(ROOT/'scripts'/name)],cwd=ROOT,check=False).returncode
        with ThreadPoolExecutor(max_workers=2) as pool:statuses=list(pool.map(summarize,['summarize_morpho_confidence_cave.py','summarize_morpho_confidence_k17.py']))
        ledger=json.loads((BASE/'jobs.json').read_text());ledger['summary_exit_codes']=statuses
        missing=[j['report'] for j in plan['jobs']+plan['reused'] if not (ROOT/j['report']).exists()]
        ledger['missing_final_reports']=missing;ledger['all_150_reports_present']=not missing
        (BASE/'jobs.json').write_text(json.dumps(ledger,indent=2)+'\n')
        if missing or any(statuses):raise RuntimeError(f'Sweep incomplete: {len(missing)} missing final reports')
        print('All 150 reports present; both five-fold summaries written',flush=True);return
    end=min(start+30,total)-1
    ledger=json.loads((BASE/'jobs.json').read_text())
    existing=next((w for w in ledger['waves'] if w.get('first_task')==start),None)
    if existing:
        print('Wave already submitted; keeping existing jobs:',json.dumps(existing),flush=True)
        return
    preempt=set(ledger.get('preemptable_task_indices',[]))
    normal_tasks=[i for i in range(start,end+1) if i not in preempt]
    preempt_tasks=[i for i in range(start,end+1) if i in preempt]
    previous_preempt = next((w.get('preemptable_training_job') for w in reversed(ledger['waves']) if w.get('preemptable_training_job')), None)
    job=submit('--array='+','.join(map(str,normal_tasks))+'%4','scripts/sbatch/train_cave_teasar_confidence_sweep.sh')
    record=dict(training_job=int(job),normal_tasks=normal_tasks,first_task=start,last_task=end,submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat());ledger['waves'].append(record)
    (BASE/'jobs.json').write_text(json.dumps(ledger,indent=2)+'\n')
    preempt_dependency = [f'--dependency=afterany:{previous_preempt}'] if previous_preempt else []
    preempt_job=submit(*preempt_dependency,'--partition=mit_preemptable','--account=mit_general','--qos=normal',
                       '--array='+','.join(map(str,preempt_tasks))+'%4','scripts/sbatch/train_cave_teasar_confidence_preemptable.sh')
    record.update(preemptable_training_job=int(preempt_job),preemptable_tasks=preempt_tasks,preemptable_dependency=preempt_dependency[0] if preempt_dependency else None)
    (BASE/'jobs.json').write_text(json.dumps(ledger,indent=2)+'\n')
    dependency = f'afterany:{job}:{preempt_job}' if end+1>=total else f'afterany:{job}'
    record['continuation_dependency'] = dependency
    continuation=submit(f'--dependency={dependency}',f'--export=ALL,CONFIDENCE_NEXT_START={end+1}','scripts/sbatch/continue_cave_teasar_confidence_sweep.sh')
    record['continuation_job']=int(continuation);(BASE/'jobs.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
