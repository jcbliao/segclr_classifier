"""Refresh cached report notebooks in parallel, without inference."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_report_fold_sweep import BASE,read,save

def render(record):
    path=ROOT/record['all_folds'];log=ROOT/'logs'/('render_'+path.stem+'.log')
    with log.open('w') as stream:
        code=subprocess.call([sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute','--inplace',
            '--ExecutePreprocessor.timeout=1200','--ExecutePreprocessor.kernel_name=segclr_db',str(path)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if code==0 and record.get('original'):
        (ROOT/record['original']).write_bytes(path.read_bytes())
    return {'notebook':record['all_folds'],'exit_code':code,'log':str(log.relative_to(ROOT))}

def main(final=False):
    records=read(BASE/'notebooks.json')
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(render,records))
    plan=read(BASE/'plan.json');missing=[j['report'] for j in plan['jobs'] if not (ROOT/j['report']).exists()]
    structure=ROOT/'analysis/all_windows/structure_ablation'
    expected=[folder/'cache'/name for folder in [structure]+[structure/f'fold{f}' for f in range(1,5)]
              for name in ['ablations_seed0.json','ablations_seed1.json','ablations_seed2.json','weight_analysis.json']]
    expected += [BASE/f'cache/augmentation_propagation_fold{fold}.json' for fold in range(5)]
    missing_inference=[str(p.relative_to(ROOT)) for p in expected if not p.exists()]
    save(BASE/('final_status.json' if final else 'notebook_status.json'),{'notebooks':results,'missing_training_reports':missing,
        'training_runs':len(plan['jobs']),'all_training_reports_present':not missing,'missing_inference_caches':missing_inference})
    print(json.dumps({'notebooks':len(results),'failed_notebooks':sum(r['exit_code']!=0 for r in results),'missing_training_reports':len(missing)}),flush=True)
    if any(r['exit_code']!=0 for r in results) or (final and (missing or missing_inference)):raise SystemExit(1)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');main(ap.parse_args().final)
