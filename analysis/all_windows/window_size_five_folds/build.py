"""Freeze missing runs using existing report settings and five-fold manifests."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.train_gnn import build_training_parser
from scripts.training_config import parse_training_args
from data.packing_cache import identity,ready

if (BASE/'jobs.json').exists():
    raise RuntimeError('Already submitted; do not rewrite frozen configs')
plan=json.loads((ROOT/'analysis/report_folds_20261012/plan.json').read_text())
group=next(g for g in plan['groups'] if g['category']=='Whole-cell linear versus ResNet')
manifests={f:json.loads((ROOT/group['runs'][f]['config']).read_text())['manifest'] for f in range(1,5)}
stems=['mean_resnet4x128','pointwise_mlp_L2_resnet4x128','mpnn_L2_resnet4x128','gt_L2_H4_resnet4x128']
jobs=[];packing={};runs=[]
defaults=vars(build_training_parser().parse_args([]))
checked=set()
for k in [10,20,40]:
    for stem in stems:
        template=ROOT/f'results/all_windows/gnn_lcpn_scratch_{stem}_n{k}_fold0.json'
        original=json.loads(template.read_text())
        for f in range(5):
            report=ROOT/f'results/all_windows/gnn_lcpn_scratch_{stem}_n{k}_fold{f}.json'
            runs.append(dict(model=stem,nodes=k,fold=f,report=str(report.relative_to(ROOT))))
            if report.exists():continue
            assert f>0
            a={**copy.deepcopy(defaults),**copy.deepcopy(original['args'])}
            a.update(dataset='all_windows',manifest=manifests[f],results_dir=str(report.parent),resume=True,amp=True,optimized_loading=True,cuda_graphs=True,num_workers=15)
            for field in ['selection_domain','embedding_augmentation']:
                if a.get(field) is None:a.pop(field,None)
            cp=BASE/'configs'/f'{len(jobs):03d}.json';cp.write_text(json.dumps(a,indent=2)+'\n')
            parsed=parse_training_args(build_training_parser(),['--config',str(cp)])
            m=json.loads(Path(parsed.manifest).read_text());key=identity(parsed,m)
            if k not in checked:
                for rid in m['cells']:
                    assert (Path(parsed.neighborhood_root)/'neighborhoods'/f'n{k}'/f'{rid}.npz').exists(),(k,rid)
                checked.add(k)
            item=dict(index=len(jobs),model=stem,nodes=k,fold=f,config=str(cp.relative_to(ROOT)),report=str(report.relative_to(ROOT)),template=str(template.relative_to(ROOT)),packing_key=key)
            jobs.append(item)
            packing.setdefault(key,dict(index=len(packing),key=key,config=item['config'],nodes=k,fold=f,ready=ready(key)))
assert len(jobs)==40
out=dict(description='Missing folds for matched 10/20/40-node whole-cell ResNet-head comparisons; preserve existing reports and original training settings.',jobs=jobs,packing=list(packing.values()),runs=runs,prediction_root='/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/window_size_five_folds/predictions')
(BASE/'plan.json').write_text(json.dumps(out,indent=2)+'\n')
print('Validated',len(jobs),'missing training runs;',len(packing),'shared packing groups;',sum(p['ready'] for p in packing.values()),'already packed',flush=True)
