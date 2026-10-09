"""Five cell-disjoint folds for the existing geodesic 10-um ResNet probe."""
import argparse,json,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO.parent/'subcompartment_classification'
DATA=Path('/orcd/scratch/orcd/013/jcbliao/segclr/subcompartment_linear_v1')
OUT=REPO/'results/subcompartment_geodesic_10um_cv'

def prepare():
 cells={}
 for folder in [DATA/'soma_cells',DATA/'geodesic/cells']:
  for path in sorted(folder.glob('*.npz')):
   with np.load(path) as z:
    y=z['y']
    if not len(y):continue
    root=str(z['root_id']);split=str(z['split'])
    item=cells.setdefault(root,dict(split=split,counts=np.zeros(4,np.int64)))
    assert item['split']==split
    item['counts']+=np.bincount(y.astype(np.int64),minlength=4)
 manifest=json.loads((REPO/'data/manifest.json').read_text())['cells']
 assigned={r:0 for r,v in cells.items() if v['split']!='train'}
 strata=defaultdict(list)
 for root,item in cells.items():
  if item['split']=='train':strata[manifest[root]['cell_type']].append(root)
 rng=np.random.default_rng(0);loads=np.zeros(4,np.int64)
 for kind,roots in sorted(strata.items(),key=lambda item:(-len(item[1]),item[0])):
  roots=sorted(roots);rng.shuffle(roots)
  # Balance each cell type across four remaining folds, then total cell counts.
  within=np.zeros(4,np.int64)
  for root in roots:
   fold=min(range(4),key=lambda i:(within[i],loads[i],i))
   assigned[root]=fold+1;within[fold]+=1;loads[fold]+=1
 assert set(assigned)==set(cells)
 summaries=[]
 for fold in range(5):
  test=[r for r in cells if assigned[r]==fold];train=[r for r in cells if assigned[r]!=fold]
  train_counts=sum((cells[r]['counts'] for r in train),np.zeros(4,np.int64))
  test_counts=sum((cells[r]['counts'] for r in test),np.zeros(4,np.int64))
  assert not set(train)&set(test) and np.all(train_counts>0) and np.all(test_counts>0)
  summaries.append(dict(fold=fold,n_train_cells=len(train),n_test_cells=len(test),n_train_points=int(train_counts.sum()),n_test_points=int(test_counts.sum()),train_counts=train_counts.tolist(),test_counts=test_counts.tolist()))
 assert summaries[0]['n_train_cells']==1828 and summaries[0]['n_test_cells']==457
 assert summaries[0]['n_train_points']==9685246 and summaries[0]['n_test_points']==2417248
 plan=dict(data=str(DATA),source_trainer=str(SOURCE/'train_compartment_gpu.py'),trainer_sha256=hashlib.sha256((SOURCE/'train_compartment_gpu.py').read_bytes()).hexdigest(),seed=0,n_folds=5,assignments=assigned,summaries=summaries,fold0_model=str(DATA/'models_resnet_inverse_sqrt_100epochs/geodesic/model.pt'),selection_policy='Match existing model: best macro F1 on held-out fold over 100 epochs. Fold metrics are used for checkpoint selection, not independent final evaluation.')
 OUT.mkdir(parents=True,exist_ok=True)
 dest=OUT/'folds.json'
 if dest.exists():assert json.loads(dest.read_text())==plan,'Existing plan differs'
 else:dest.write_text(json.dumps(plan,indent=2)+'\n')
 print(json.dumps(summaries,indent=2),flush=True)

def train(fold):
 assert 1<=fold<=4
 plan=json.loads((OUT/'folds.json').read_text())
 assert hashlib.sha256((SOURCE/'train_compartment_gpu.py').read_bytes()).hexdigest()==plan['trainer_sha256']
 sys.path.insert(0,str(SOURCE))
 import train_compartment_gpu as gpu
 original=gpu.load
 def load(variant):
  x,y,roots,old_train,root_ids=original(variant)
  assert set(map(str,root_ids))==set(plan['assignments'])
  cell_folds=np.array([plan['assignments'][str(root)] for root in root_ids])
  mask=cell_folds[roots]!=fold
  return x,y,roots,mask,root_ids
 gpu.load=load;gpu.BASE=OUT/f'fold{fold}'
 gpu.run('geodesic',epochs=100,head='resnet',sampling='inverse_sqrt')
 folder=gpu.BASE/'models_resnet_inverse_sqrt_100epochs/geodesic'
 checkpoint=gpu.torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
 checkpoint.update(fold=fold,n_folds=5,selection_split=f'held-out cell fold {fold}',fold_plan=str(OUT/'folds.json'))
 gpu.torch.save(checkpoint,folder/'model.pt')
 for name in ['training.json','metrics.json']:
  path=folder/name;d=json.loads(path.read_text())
  d.update(fold=fold,n_folds=5,fold_plan=str(OUT/'folds.json'),selection_split=f'held-out cell fold {fold}',test_policy=plan['selection_policy'])
  path.write_text(json.dumps(d,indent=2)+'\n')
 print('Fold complete',fold,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--fold',type=int)
 a=p.parse_args()
 if a.prepare:prepare()
 elif a.fold is not None:train(a.fold)
 else:p.error('Specify --prepare or --fold')
