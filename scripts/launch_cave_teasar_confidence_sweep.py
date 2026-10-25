"""Freeze and audit matched five-fold CAVE/TEASAR confidence experiments."""
import hashlib,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.dataset_lcpn import load_hierarchy
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010'
DATA=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons')
CONFIDENCES=['0','0.3','0.5','0.7','0.9']
FAMILIES={'CAVE':('morpho_confidence_cave/k10','results/presynaptic/morpho_confidence_cave/k10',10),
          'TEASAR':('casey_coarse_confidence_with_tc_folds/scale16/k17','results/presynaptic/morpho_coarse_confidence_with_tc_mixed16_stratified5/scale16/k17',17)}
TAGS={'mean':'mean','pointwise_mlp':'pointwise_mlp_L2','graph_transformer':'gt_L4_H4'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    jobs=[];reused=[];cohorts=[];seen={};roots={f:set() for f in FAMILIES};all_manifests={}
    for c in CONFIDENCES:
        for fold in range(5):
            paired={}
            for family,(data_sub,res_sub,k) in FAMILIES.items():
                db=DATA/data_sub/f'conf{c}'/f'fold{fold}';mp=db/'manifest.json';m=json.loads(mp.read_text());metadata=json.loads((db/'metadata.json').read_text())
                assert metadata['k_observed']==k
                if family=='TEASAR':assert metadata['format']=='dense-presynaptic-v1' and not metadata['missing_root_ids']
                else:assert metadata['format']=='presynaptic-axon-npz-v1'
                h=load_hierarchy(m);assert h.level_classes[-1]==['BasketFam','BipFam','MartFam','NglFam','pyramidal','thalamocortical']
                assert m.get('fold_index',m.get('split_seed'))==fold
                paired[family]={rid:(v['cell_type'],v['split']) for rid,v in m['cells'].items()}
                all_manifests[(family,c,fold)]=m
                for rid,v in m['cells'].items():
                    if v['split']=='test':
                        key=(family,rid)
                        assert key not in seen or seen[key]==fold
                        seen[key]=fold
                roots[family].update(m['cells'])
                frozen=BASE/'manifests'/f'{family}_conf{c}_fold{fold}.json';frozen.write_text(json.dumps(m,indent=2)+'\n')
                cohorts.append(dict(family=family,confidence=c,fold=fold,cells=len(m['cells']),manifest=str(frozen.relative_to(ROOT)),sha256=sha(frozen)))
                for arch,tag in TAGS.items():
                    run=f'gnn_lcpn_scratch_{tag}_resnet4x128_n{k}_mixed16_sampled_fold{fold}'
                    report=ROOT/res_sub/f'conf{c}'/f'fold{fold}'/f'{run}.json'
                    template=ROOT/res_sub/f'conf{c}'/'fold0'/f'gnn_lcpn_scratch_{tag}_resnet4x128_n{k}_mixed16_sampled_fold0.json'
                    d=json.loads(template.read_text());a=d['args'].copy()
                    assert d['classes']==h.level_classes[-1] and a['class_balance']=='sample' and a['epochs']==30 and not a['freeze_aggregator']
                    a.update(presynaptic_database=str(db),manifest=str(frozen),results_dir=str(report.parent.relative_to(ROOT)),num_workers=15,resume=True)
                    record=dict(family=family,confidence=c,fold=fold,architecture=arch,report=str(report.relative_to(ROOT)),template=str(template.relative_to(ROOT)),template_sha256=sha(template),manifest_sha256=sha(frozen))
                    if report.exists():
                        old=json.loads(report.read_text());assert old['classes']==h.level_classes[-1]
                        ignored={'manifest','num_workers','results_dir','presynaptic_database','resume'}
                        assert all(a[key]==value for key,value in old['args'].items() if key not in ignored)
                        record['report_sha256']=sha(report);reused.append(record)
                    else:
                        config=dict(a)
                        if config.get('embedding_augmentation') is None:config.pop('embedding_augmentation',None)
                        record['index']=len(jobs);cp=BASE/'configs'/f'{len(jobs):03d}.json';cp.write_text(json.dumps(config,indent=2)+'\n');record['config']=str(cp.relative_to(ROOT));record['config_sha256']=sha(cp);jobs.append(record)
            assert paired['CAVE']==paired['TEASAR'],(c,fold,'Mismatched roots, labels or split')
    for family in FAMILIES:
        for fold in range(5):
            prior=set(all_manifests[(family,'0',fold)]['cells'])
            for c in CONFIDENCES[1:]:
                current=set(all_manifests[(family,c,fold)]['cells']);assert current<=prior;prior=current
        for c in CONFIDENCES:
            m=all_manifests[(family,c,0)];held=[{rid for rid,v in all_manifests[(family,c,f)]['cells'].items() if v['split']=='test'} for f in range(5)]
            assert set.union(*held)==set(m['cells']) and sum(map(len,held))==len(m['cells'])
    def ready(item):
        family,rid=item;data_sub,_,_=FAMILIES[family];db=DATA/data_sub/'conf0/fold0'
        assert (db/'cells'/f'{rid}.npz').is_file(),(family,rid)
        if family=='TEASAR':assert (DATA/'casey_k17_memmap/scale16/k17'/rid/'complete.json').is_file(),('missing memmap',rid)
    with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(ready,[(f,r) for f,rs in roots.items() for r in rs]))
    result=dict(confidences=CONFIDENCES,folds=list(range(5)),models=list(TAGS),total_combinations=len(jobs)+len(reused),new_jobs=len(jobs),reused_jobs=len(reused),jobs=jobs,reused=reused,cohorts=cohorts,matched_cohorts=True,stable_nested_folds=True,all_input_cells_present=True)
    (BASE/'plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'jobs','reused','cohorts'}}),flush=True)
if __name__=='__main__':main()
