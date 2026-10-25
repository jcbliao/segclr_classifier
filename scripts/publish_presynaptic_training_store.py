"""Publish current prepared training inputs as a lab-readable compute bundle.

Source stores remain unchanged. Internal symlinks become relative, external
file dependencies are materialized, and only owned destination files receive
lab-group permissions. Runtime packing caches remain disposable.
"""
import argparse,hashlib,json,os,shutil,stat,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SOURCE=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons_latest100k')
FEATURE_WORK=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/latest100k_morpho6_centers_v1')
DEST=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/training_stores/latest100k_morpho6_conf0_v1')
STAGE=DEST.with_name(DEST.name+'.incomplete')
STATE=ROOT/'analysis/db_deployment/presynaptic/latest_segclr'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)

def copy_file(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if sha(source)!=sha(dest):raise ValueError(f'Existing dependency differs: {dest}')
 else:
  temporary=dest.with_suffix(dest.suffix+'.copying');shutil.copyfile(source,temporary);temporary.replace(dest)
 return sha(dest)

def copy_tree(source,dest):
 """Preserve internal references and materialize external file links."""
 dest.mkdir(parents=True,exist_ok=True)
 for entry in sorted(source.iterdir()):
  if entry.name in {'conf0.7','__pycache__'} or entry.name.endswith(('.lock','.tmp','.tmp.parquet')):continue
  target=dest/entry.name
  if entry.is_symlink():
   resolved=entry.resolve(strict=True)
   try:relative=resolved.relative_to(SOURCE)
   except ValueError:
    if resolved.is_dir():raise ValueError(f'Unexpected external directory link: {entry}')
    copy_file(resolved,target)
   else:
    final_target=DEST/relative
    link=os.path.relpath(final_target,DEST/target.parent.relative_to(STAGE))
    if not target.is_symlink():target.symlink_to(link,target_is_directory=resolved.is_dir())
  elif entry.is_dir():copy_tree(entry,target)
  elif entry.is_file():
   # Rewritten JSON and cell files are not overwritten on a resumed build.
   if not target.exists():copy_file(entry,target)


def build():
 DEST.parent.mkdir(parents=True,exist_ok=True)
 if DEST.exists():
  receipt=json.loads((DEST/'complete.json').read_text())
  if not receipt['complete']:raise ValueError('Published store requires verification before reuse')
  print('ALREADY PUBLISHED',DEST,flush=True);return
 STAGE.mkdir(exist_ok=True)
 save(STAGE/'complete.json',dict(complete=False,source=str(SOURCE),destination=str(DEST)))
 print('copying prepared window/feature stores',flush=True)
 copy_tree(SOURCE,STAGE)
 print('materializing original postsynaptic site inputs',flush=True)
 copy_tree(FEATURE_WORK/'training_post',STAGE/'postsynaptic_source')
 dependencies={};rewritten=0
 # Canonical cells are visited once; fold-level cells are relative symlinks.
 for relative in ['casey_confidence_cave/k10/cells','casey_coarse_confidence_with_tc_folds/scale16/k17/cells']:
  for index,path in enumerate(sorted((STAGE/relative).glob('*.npz'))):
   with np.load(path,allow_pickle=False) as z:
    references={key:str(z[key]) for key in ['geometry_path','embedding_path'] if key in z}
   if not references:continue
   updated={}
   for key,value in references.items():
    old=Path(value)
    # Resumed files already name their final dependency locations.
    if old.is_relative_to(DEST):continue
    if old.is_relative_to(SOURCE):
     final=DEST/old.relative_to(SOURCE)
     if not (STAGE/final.relative_to(DEST)).exists():raise ValueError(f'Missing internal dependency: {old}')
    else:
     resolved=old.resolve(strict=True)
     token=hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
     final=DEST/'geometry'/f'{token}_{resolved.name}'
     if str(resolved) not in dependencies:
      digest=copy_file(resolved,STAGE/final.relative_to(DEST))
      dependencies[str(resolved)]=dict(path=str(final),sha256=digest,bytes=resolved.stat().st_size)
    updated[key]=np.asarray(str(final))
   if updated:
    with np.load(path,allow_pickle=False) as z:arrays={key:z[key].copy() for key in z.files}
    arrays.update(updated);temporary=path.with_suffix('.publish.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path);rewritten+=1
   if index%200==0:print('rewritten native references',index,flush=True)
 # Save a resumable dependency catalog independent of the final receipt.
 previous=STAGE/'dependencies.json'
 old=json.loads(previous.read_text()) if previous.exists() else {}
 dependencies=dict(old,**dependencies);save(previous,dependencies)
 replacements={str(SOURCE):str(DEST),str(FEATURE_WORK/'training_post'):str(DEST/'postsynaptic_source')}
 def rewrite(value):
  if isinstance(value,str):
   for before,after in replacements.items():
    if value.startswith(before):return after+value[len(before):]
   return value
  if isinstance(value,list):return [rewrite(item) for item in value]
  if isinstance(value,dict):return {key:rewrite(item) for key,item in value.items()}
  return value
 for path in STAGE.rglob('*.json'):
  if path in {STAGE/'complete.json',STAGE/'dependencies.json'}:continue
  save(path,rewrite(json.loads(path.read_text())))
 plan=json.loads((STATE/'morpho6/plan.json').read_text());configs=[]
 for model in plan['models']:
  source_config=json.loads(Path(model['config']).read_text())
  source_manifest=Path(source_config['manifest'])
  relative_manifest=Path('manifests')/model['family']/f"fold{model['fold']}.json"
  save(STAGE/relative_manifest,rewrite(json.loads(source_manifest.read_text())))
  config=rewrite(source_config);config['manifest']=str(DEST/relative_manifest)
  config['results_dir']=str(Path('results/shared_latest100k')/model['family']/f"fold{model['fold']}")
  # These options are unused for the published presynaptic datasets.
  if config.get('embedding_augmentation') in (None,'clean'):config.pop('embedding_augmentation_database',None)
  config.pop('neighborhood_root',None)
  config_path=Path('configs')/f"{model['family']}_fold{model['fold']}.json"
  save(STAGE/config_path,config);configs.append(dict(family=model['family'],fold=model['fold'],config=str(DEST/config_path)))
 save(STAGE/'training_recipes.json',dict(models=configs,embedding_run_id='resnet50_1280b_slab1280_50um_mac_partial_aug_2_style_bf16_100k__20261022_140504',embedding_checkpoint_id='checkpoint_e0_s100000',cells=2029,label_confidence_min=0))
 (STAGE/'train.sh').write_text('''#!/bin/bash
set -euo pipefail
bundle=$(cd -- "$(dirname -- "$0")" && pwd)
family=${1:?Usage: train.sh cave_n10\|single_pre_post fold}
fold=${2:?Specify fold 0 through 4}
case "$family" in cave_n10|single_pre_post) ;; *) echo "Unknown family" >&2; exit 1;; esac
case "$fold" in 0|1|2|3|4) ;; *) echo "Unknown fold" >&2; exit 1;; esac
cd "${GNN_CLASSIFIER_REPO:-$PWD}"
[[ -f scripts/train_gnn.py ]] || { echo 'Run from your gnn_classifier checkout or set GNN_CLASSIFIER_REPO' >&2; exit 1; }
export SEGCLR_PACKING_CACHE="${SEGCLR_PACKING_CACHE:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/segclr_packing_${USER}}"
exec "${PYTHON:-python}" scripts/train_gnn.py --config "$bundle/configs/${family}_fold${fold}.json"
''')
 (STAGE/'README.md').write_text('''# Shared newest-SegCLR presynaptic training store

Prepared inputs for CAVE K10 pre/post and native single pre/post classifiers,
five folds, 2,029 cells, six morphological classes, no label-confidence cutoff.
Windows, selected centers and source geometry are unchanged. Features use
`checkpoint_e0_s100000` from the 2026-10-22 SegCLR run.

All windows, node embeddings, native geometry dependencies, postsynaptic caches,
raw postsynaptic site inputs, manifests and ten training recipes live here.
The bundle is readable by `orcd_rg_fstor012_pi_sdorkenw`. It requires neither
traversal of Jacob's home/scratch directories nor live database access to train
from these prepared inputs. Historical provenance may still name source paths.

## Train

Use your own clone of https://github.com/jcbliao/segclr_classifier and an
environment with its training dependencies (PyTorch/PyG and the hierarchy
package). Run on an allocated GPU from your checkout:

```bash
PYTHON=/path/to/your/environment/bin/python \\
  '''+str(DEST)+'''/train.sh cave_n10 0
```

The other family is `single_pre_post`; folds are 0 through 4. Results go into
`results/shared_latest100k` relative to your checkout. `GNN_CLASSIFIER_REPO`
can explicitly select the checkout. `SEGCLR_PACKING_CACHE` can select a writable
runtime packing location; otherwise it uses the job's temporary directory.
Recent training code removes packed runtime caches after successful use.
The published prepared windows/features remain available here.
Postsynaptic vectors live once in `postsynaptic_features/`; CAVE and native
caches hold their own alignment maps and eligibility. Use a checkout containing
`data/postsynaptic_cache.py` to read this shared format.

Native inputs contain center-only embeddings for the single-input model and
retain their full geometry references. They do not provide newest embeddings
for every unused native vertex. Raw source sites are in `postsynaptic_source`.
`complete.json` records verification; `dependencies.json` records copied geometry.
`training_recipes.json` lists the exported configurations.
''')
 from scripts.share_postsynaptic_features import share
 share(STAGE)
 # Apply permissions only to this owned publication tree.
 group=DEST.parent.stat().st_gid
 for directory,dirs,files in os.walk(STAGE,followlinks=False):
  path=Path(directory);os.chown(path,-1,group);os.chmod(path,0o2750)
  for name in dirs+files:
   path=Path(directory)/name;os.chown(path,-1,group,follow_symlinks=False)
   if not path.is_symlink():os.chmod(path,0o2750 if path.is_dir() else (0o750 if path.name=='train.sh' else 0o640))
 STAGE.rename(DEST)
 validate(dependencies,configs)


def validate(dependencies=None,configs=None):
 from data.dataset_presynaptic import load_cell_arrays
 from scripts.train_gnn import build_training_parser
 from scripts.training_config import parse_training_args
 recipes=json.loads((DEST/'training_recipes.json').read_text())
 for model in recipes['models']:
  args=parse_training_args(build_training_parser(),['--config',model['config']])
  for value in [args.manifest,args.presynaptic_database,args.postsynaptic_cache]:
   path=Path(value)
   if not path.is_relative_to(DEST) or not path.exists():raise ValueError(f'Nonportable training dependency: {value}')
  marker=json.loads((Path(args.postsynaptic_cache)/'complete.json').read_text())
  for field in ['database','embeddings']:
   path=Path(marker[field])
   if not path.is_relative_to(DEST) or not path.exists():raise ValueError(f'Nonportable postsynaptic cache metadata: {field}/{path}')
 # Every embedded geometry/vector reference and every symlink stays within bundle.
 references=0;files=0;bytes_total=0
 for directory,dirs,names in os.walk(DEST,followlinks=False):
  for name in dirs+names:
   path=Path(directory)/name
   if path.is_symlink() and (not path.resolve().is_relative_to(DEST) or not path.exists()):raise ValueError(f'Nonportable link: {path}')
  for name in names:
   path=Path(directory)/name
   if path.is_symlink():continue
   files+=1;bytes_total+=path.stat().st_size
   if path.suffix=='.npz' and path.parent.name=='cells':
    with np.load(path,allow_pickle=False) as z:
     for key in ['geometry_path','embedding_path']:
      if key not in z:continue
      ref=Path(str(z[key]));references+=1
      if not ref.is_relative_to(DEST) or not ref.exists():raise ValueError(f'Nonportable cell dependency: {path}/{key}')
   if stat.S_IMODE(path.stat().st_mode)&0o040==0:raise ValueError(f'File lacks group readability: {path}')
 for relative,variant,single in [('casey_confidence_cave/k10','cave',False),('casey_coarse_confidence_with_tc_folds/scale16/k17','new',True)]:
  original=json.loads((SOURCE/relative/'conf0/fold0/manifest.json').read_text())
  published=json.loads((DEST/relative/'conf0/fold0/manifest.json').read_text())
  if original['cells']!=published['cells']:raise ValueError('Cohort/fold changed')
  for split in ['train','test']:
   rid=next(r for r,info in original['cells'].items() if info['split']==split)
   before=load_cell_arrays(SOURCE/relative/'conf0/fold0/cells'/f'{rid}.npz',variant,single)
   after=load_cell_arrays(DEST/relative/'conf0/fold0/cells'/f'{rid}.npz',variant,single)
   if before.keys()!=after.keys():raise ValueError('Cell schema changed')
   for key in before:np.testing.assert_array_equal(before[key],after[key])
 receipt=dict(complete=True,source=str(SOURCE),destination=str(DEST),files=files,bytes=bytes_total,rewritten_references=references,cells=2029,configurations=2,folds=5,group=DEST.stat().st_gid,checks=['All ten configurations parse and reference shared inputs','All cell references and symlinks resolve inside bundle','All files group readable','Cohort labels and fold assignments unchanged','Native/CAVE train/test sample arrays identical to originals'],geometry_dependencies=len(json.loads((DEST/'dependencies.json').read_text())))
 save(DEST/'complete.json',receipt);os.chmod(DEST/'complete.json',0o640)
 save(STATE/'shared_training_store.json',receipt)
 print('SHARED TRAINING STORE VERIFIED',json.dumps(receipt),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['build','verify']);args=parser.parse_args()
 build() if args.stage=='build' else validate()
