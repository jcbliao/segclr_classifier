"""Freeze report training runs, preserving historical reports and fold zero."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,random,re,sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.dataset_lcpn import load_hierarchy
from scripts.train_gnn import build_training_parser
from scripts.training_config import parse_training_args
BASE=ROOT/'analysis/report_folds_20261012'
DATA=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
DENSITY=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/report_folds_20261012/density')
TAGS={'mean':'mean','pointwise_mlp':'pointwise_mlp_L2','graph_transformer':'gt_L4_H4'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp.json');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def assignments(manifest):
    """Retain original held-out fold 0; partition remaining cells by fine label."""
    result={rid:0 for rid,info in manifest['cells'].items() if info['split']=='test'}
    strata=defaultdict(list)
    for rid,info in manifest['cells'].items():
        if rid not in result:strata[info['cell_type']].append(rid)
    for label,roots in sorted(strata.items()):
        rng=random.Random(int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'big'))
        roots=sorted(roots);rng.shuffle(roots)
        order=[1,2,3,4];rng.shuffle(order)
        result.update({rid:order[i%4] for i,rid in enumerate(roots)})
    assert len(result)==len(manifest['cells'])
    return result
def folded(source,fold,mapping=None):
    mapping=assignments(source) if mapping is None else mapping
    cells={rid:{**info,'split':'test' if mapping[rid]==fold else 'train'} for rid,info in source['cells'].items()}
    return {**source,'cells':cells,'fold_index':fold,'n_folds':5,
            'report_fold_policy':'historical held-out fold 0 retained; remaining roots stratified across folds 1-4'}

def build(rerun_baselines=False):
    if (BASE/'jobs.json').exists():raise RuntimeError('Already submitted; refusing to rewrite frozen plan')
    audit=read(ROOT/'analysis/report_audit_20261012/fold_audit.json')
    original_whole=read(ROOT/'analysis/all_windows/linear_vs_resnet/manifest.json')
    historical_map=assignments(original_whole)
    morpho={f:read(DATA/f'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold{f}/manifest.json') for f in range(5)}
    vbase=read(DATA/'cave_v1dd_addition/manifests/with_v1dd/fold0.json')
    vfold={r['root_id']:int(r['held_out_fold']) for r in csv.DictReader((DATA/'cave_v1dd_addition/v1dd_fold_assignments.csv').open())}
    vman={}
    for f in range(5):
        mic=read(DATA/f'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0/fold{f}/manifest.json')
        cells={rid:{**info,'split':mic['cells'][rid]['split']} if info['source_dataset']=='microns'
               else {**info,'split':'test' if vfold[rid]==f else 'train'} for rid,info in vbase['cells'].items()}
        for arm in ['no_v1dd','with_v1dd']:
            vman[f,arm]={**vbase,'fold_index':f,'training_condition':arm,'cells':{
                rid:info for rid,info in cells.items() if arm=='with_v1dd' or info['source_dataset']=='microns' or info['split']=='test'}}
    groups=[];jobs=[]
    for index,old in enumerate(audit['groups']):
        category=old['category'];existing=old['reports'];template=existing[0] if existing else None
        if not old['missing_folds']:
            groups.append({'id':index,'category':category,'runs':[{'fold':f,'report':next(p for p in existing if f'fold{f}' in p),
                'new':False,'report_sha256':sha(ROOT/next(p for p in existing if f'fold{f}' in p))} for f in range(5)]})
            continue
        if template is None:
            # CAVE mean / GT pre+post: preserve the existing six-class post cohort.
            template=next(g['reports'][0] for g in audit['groups'] if g['category']==category and g['reports'])
        payload=read(ROOT/template);a0={**vars(build_training_parser().parse_args([])),**copy.deepcopy(payload['args'])}
        if not existing:
            arch='graph_transformer' if '_gt_' in old['report_pattern'] else 'mean'
            reference=read(next((ROOT/'results/presynaptic/morpho_confidence_cave/k10/conf0.7/fold0').glob(f'gnn_lcpn_scratch_{TAGS[arch]}_*.json')))
            a0={**reference['args'],**{k:v for k,v in a0.items() if k not in ['architecture']}}
            a0['architecture']=arch
        group={'id':index,'category':category,'template':template,'template_sha256':sha(ROOT/template),'runs':[]}
        source_path=a0.get('manifest') or (str(Path(a0['presynaptic_database'])/'manifest.json') if a0['dataset']!='all_windows' else str(ROOT/'analysis/all_windows/linear_vs_resnet/manifest.json'))
        source=read(source_path)
        for f in range(5):
            a=copy.deepcopy(a0)
            target=re.sub(r'fold\[0-4\]',f'fold{f}',old['report_pattern'])
            rerun_v=category=='V1DD addition' and f==0 and ('/with_v1dd/' in target or rerun_baselines)
            density=category=='Historical TEASAR density sweep'
            new=density or f in old['missing_folds'] or rerun_v
            if density:
                scale=re.search(r'/scale(\d+)/',target)[1];k=re.search(r'/k(\d+)/',target)[1]
                target=f'results/presynaptic/morpho_density_20261012/conf0.7/scale{scale}/k{k}/fold{f}/'+Path(target).name
                a['presynaptic_database']=str(DENSITY/f'scale{scale}/k{k}')
                a['presynaptic_memmap_root']=None
                manifest=copy.deepcopy(morpho[f]);manifest['source_database']=a['presynaptic_database']
                group['category']='Morpho TEASAR density sweep'
            elif category=='Whole-cell linear versus ResNet':
                manifest=folded(original_whole,f,historical_map)
            elif category=='Historical CAVE versus original mesh TEASAR':
                mapping=assignments(source)
                mapping.update({rid:historical_map[rid] for rid in mapping if rid in historical_map})
                assert all((mapping[rid]==0)==(info['split']=='test') for rid,info in source['cells'].items())
                manifest=folded(source,f,mapping)
            elif category=='V1DD addition':
                arm='with_v1dd' if '/with_v1dd/' in target else 'no_v1dd'
                manifest=vman[f,arm]
                if new:
                    target=f'results/presynaptic/cave_v1dd_addition/microns_selection/{arm}/'+Path(target).name
                    a['selection_domain']='microns'
            elif category in ['TEASAR neighborhood pre/post','Single-embedding pre/post','CAVE six-class pre+post']:
                a['presynaptic_database']=re.sub(r'fold\d+',f'fold{f}',a['presynaptic_database'])
                manifest=read(Path(a['presynaptic_database'])/'manifest.json')
            else:
                original=next(p for p in existing if f'fold{f}' in p)
                target=original;manifest=None
            record={'fold':f,'report':target,'new':new}
            if new:
                mp=BASE/'manifests'/f'group{index:02d}_fold{f}.json';save(mp,manifest)
                a.update(manifest=str(mp),results_dir=str((ROOT/target).parent),num_workers=15,
                         resume=True,amp=True,optimized_loading=True,cuda_graphs=True)
                if a.get('embedding_augmentation') is None:a.pop('embedding_augmentation',None)
                if a.get('selection_domain') is None:a.pop('selection_domain',None)
                # A template may contain historical paths in its old results_dir only.
                assert a.get('freeze_aggregator',False) is False
                cp=BASE/'configs'/f'{len(jobs):03d}.json';save(cp,a)
                parsed=parse_training_args(build_training_parser(),['--config',str(cp)])
                h=load_hierarchy(manifest)
                if density:assert h.level_classes[-1]==['BasketFam','BipFam','MartFam','NglFam','pyramidal','thalamocortical']
                record.update(index=len(jobs),config=str(cp.relative_to(ROOT)),config_sha256=sha(cp),manifest_sha256=sha(mp),
                              template=template,template_sha256=sha(ROOT/template),category=group['category'])
                if density:record['requires']=str(DENSITY/'complete.json')
                jobs.append(record.copy())
            else:
                d=read(ROOT/target);assert 'window_test_metrics' in d
                record['report_sha256']=sha(ROOT/target)
            group['runs'].append(record)
        groups.append(group)
    expected=204 if rerun_baselines else 201
    assert len(jobs)==expected,(len(jobs),expected)
    # Audit every input root concurrently; no remote queries or geometry decoding.
    tasks=set()
    for job in jobs:
        a=read(ROOT/job['config']);m=read(a['manifest'])
        h=load_hierarchy(m)
        if a['dataset']!='all_windows':
            from data.dataset_presynaptic import NON_NEURON_LABELS
            from gnn.hierarchy import with_dropped_labels
            h=with_dropped_labels(h,NON_NEURON_LABELS)
        from data.dataset_lcpn import split_cells
        eligible=dict(split_cells(m,'train',h)+split_cells(m,'test',h))
        for rid in eligible:
            if a['dataset']=='all_windows':
                tasks.add(str(ROOT/f'data/graph_cache/{rid}.pt'))
                tasks.add(str(Path(a['neighborhood_root'])/f'neighborhoods/n10/{rid}.npz'))
            else:
                path=Path(a['presynaptic_database'])/f'cells/{rid}.npz'
                if str(path).startswith(str(DENSITY)) and not path.is_file():
                    relative=path.relative_to(DENSITY)
                    historical=DATA/'resampled_teasar_2209/training_local_center'/relative
                    path=historical if historical.is_file() else DATA/f'casey_k17_full_tc_source/scale16/k17/cells/{rid}.npz'
                tasks.add(str(path))
                if a.get('presynaptic_memmap_root'):tasks.add(str(Path(a['presynaptic_memmap_root'])/str(rid)/'complete.json'))
                if a.get('postsynaptic_cache'):tasks.add(str(Path(a['postsynaptic_cache'])/f'cells/{rid}.npz'))
    with ThreadPoolExecutor(max_workers=16) as pool:
        missing=[p for p,exists in pool.map(lambda p:(p,Path(p).is_file()),sorted(tasks)) if not exists]
    save(BASE/'input_audit.json',{'checked_files':len(tasks),'missing':missing})
    if missing:raise RuntimeError(f'{len(missing)} missing inputs; see input_audit.json')
    frozen={'jobs':jobs,'groups':groups,'new_runs':len(jobs),'rerun_v1dd_baselines':rerun_baselines,
            'density_cohort':'six-class morpho confidence >=0.7; original five-fold root assignments',
            'normal_concurrency':4,'preemptable_concurrency':4,'preemptable_indices':[j['index'] for j in jobs if j['index']%4==0],
            'input_files_checked':len(tasks),'audit_source_sha256':sha(ROOT/'analysis/report_audit_20261012/fold_audit.json')}
    save(BASE/'plan.json',frozen)
    print(json.dumps({k:v for k,v in frozen.items() if k not in ['jobs','groups','preemptable_indices']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rerun-v1dd-baselines',action='store_true');args=ap.parse_args()
    build(args.rerun_v1dd_baselines)
