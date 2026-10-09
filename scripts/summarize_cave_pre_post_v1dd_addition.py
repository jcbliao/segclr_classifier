"""Compare post-point effects with V1DD training on paired fold-0 tests."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path('results/presynaptic/cave_pre_post_v1dd_addition')
OUT=Path('analysis/presynaptic/cave_pre_post_v1dd_addition')
rows=[];counts={}
for augmentation in ['with_v1dd']:
 for feature in ['pre_only','pre_post','pre_mean_control']:
  found={}
  for p in (ROOT/augmentation/feature).glob('gnn_*.json'):
   r=json.loads(p.read_text());arch=r['args']['architecture']
   if arch in found:raise ValueError(f'Duplicate model result: {augmentation}/{feature}/{arch}')
   found[arch]=r
  assert set(found)=={'mean','pointwise_mlp','graph_transformer'},f'Incomplete results: {augmentation}/{feature}'
  for arch,r in found.items():
   assert r['use_postsynaptic']==(feature=='pre_post')
   assert r['append_presynaptic_mean']==(feature=='pre_mean_control')
   if augmentation in counts:assert counts[augmentation]==r['dataset_counts']
   else:counts[augmentation]=r['dataset_counts']
   domains={'combined':{'cell':r['test_metrics'],'window':r['window_test_metrics']},**r['dataset_test_metrics']}
   for domain,metrics in domains.items():
    row=dict(augmentation=augmentation,feature=feature,architecture=arch,dataset=domain,fold=0,best_epoch=r['best_epoch'])
    for level in ['cell','window']:
     for key in ['accuracy','balanced_accuracy','macro_precision','macro_f1']:row[level+'_'+key]=metrics[level][key]
     row['n_'+level]=sum(map(sum,metrics[level]['confusion_matrix']))
    rows.append(row)
OUT.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(OUT/'summary.csv',index=False)
keys=['augmentation','architecture','dataset','fold'];metric=[c for c in df if c.startswith(('cell_','window_'))];deltas=[]
for reference in ['pre_only','pre_mean_control']:
 pairs=df[df.feature.eq('pre_post')].merge(df[df.feature.eq(reference)],on=keys,suffixes=('_post','_reference'),validate='one_to_one')
 for _,r in pairs.iterrows():deltas.append({**{k:r[k] for k in keys},'reference':reference,**{k:r[k+'_post']-r[k+'_reference'] for k in metric}})
pd.DataFrame(deltas).to_csv(OUT/'post_deltas.csv',index=False)
print(df.to_string(index=False))
