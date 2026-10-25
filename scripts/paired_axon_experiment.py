"""Paired CAVE axon-only versus axon+dendrite experiment, no shared DB writes."""
from __future__ import annotations
import argparse, json, os, sys, hashlib, shutil, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/axon_vs_whole_cell_casey6_20261009')
SOURCE=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/morpho_confidence_cave/k10/conf0.7')
POST=Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_skeletons_pre_post/k10/conf0.7/postsynaptic')
NEIGHBOR=Path('/orcd/scratch/orcd/013/jcbliao/embedding_paths/r5um/neighborhoods/n10')
DOC=ROOT/'analysis/presynaptic/axon_vs_whole_cell_morpho6'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def stamp(p):
    p=Path(p);s=p.stat();return dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)
def frozen_load():
    plan=json.loads((BASE/'plan.json').read_text())
    for item in plan['inputs']:
        if stamp(item['path'])!=item:raise ValueError(f"Changed source: {item['path']}")
    return plan

def padded(x,offsets,members,rows):
    """Select ragged node bags without averaging before nonlinear phi."""
    rows=np.asarray(rows);counts=offsets[rows+1]-offsets[rows]
    if len(counts) and (counts.min()<1 or counts.max()>10):raise ValueError('Unexpected node count')
    mask=np.arange(10)[None,:]<counts[:,None]
    locations=offsets[rows,None]+np.arange(10)[None,:]
    nodes=np.zeros((len(rows),10),np.int64);nodes[mask]=members[locations[mask]]
    features=np.zeros((len(rows),10,64),np.float32);features[mask]=x[nodes[mask]]
    if not np.isfinite(features).all():raise ValueError('Nonfinite input')
    return features,mask

def initialize():
    from data.dataset_lcpn import load_hierarchy
    from scripts.subcompartment_predict_all import CV
    BASE.mkdir(parents=True,exist_ok=True)
    manifests=[SOURCE/f'fold{f}/manifest.json' for f in range(5)]
    sources=[json.loads(p.read_text()) for p in manifests]
    roots=sorted(sources[0]['cells'],key=int)
    for m in sources[1:]:
        assert set(m['cells'])==set(roots)
        assert all(m['cells'][r]['cell_type']==sources[0]['cells'][r]['cell_type'] for r in roots)
    h=load_hierarchy(sources[0]);assert len(h.level_classes[-1])==6
    cv=json.loads((CV/'folds.json').read_text());models=[]
    import torch
    for f in range(5):
        p=Path(cv['fold0_model']) if f==0 else CV/f'fold{f}/models_resnet_inverse_sqrt_100epochs/geodesic/model.pt'
        ck=torch.load(p,map_location='cpu',weights_only=True)
        assert ck['variant']=='geodesic' and ck['embedding_dim']==64 and ck['head']=='resnet'
        models.append(dict(fold=f,path=str(p),sha256=sha(p),classes=ck['classes']))
    inputs=[stamp(p) for p in manifests]+[stamp(ROOT/'data/nucleus_positions.json')]
    cells=[]
    for r in roots:
        p=SOURCE/'fold0/cells'/f'{r}.npz';post=POST/'cells'/f'{r}.npz'
        geom=ROOT/'data/graph_cache'/f'{r}.pt';neigh=NEIGHBOR/f'{r}.npz'
        inputs.extend([stamp(p),stamp(post)])
        dend=geom.exists() and neigh.exists()
        if dend:inputs.extend([stamp(geom),stamp(neigh)])
        label=sources[0]['cells'][r]['cell_type'];path=h.label_paths[label]
        cells.append(dict(root_id=r,cell_type=label,targets=[h.level_maps[i][v] for i,v in enumerate(path)],
                          dendritic_source_available=dend,source=str(p.resolve())))
    plan=dict(format=1,cells=cells,models=models,inputs=inputs,manifest_paths=list(map(str,manifests)),
              level_classes=h.level_classes,epochs=30,batch_size=4096,seed=0,
              class_balance='sample; inverse square root of class window count',
              split='existing outer test roots; inner validation ~10% of train roots stratified by six-class label',
              input='phi(node) mean + post on axons; phi(node) mean + raw node mean on dendrites',
              selection='per-fold subcompartment classifier of raw mean; axon/dendrite argmax',
              checkpoint_selection='validation axon-window macro F1; test only after best checkpoint reload',
              shared_database_writes=False)
    p=BASE/'plan.json'
    if p.exists() and json.loads(p.read_text())!=plan:raise ValueError('Existing frozen plan differs')
    atomic_json(p,plan);atomic_json(DOC/'plan.json',plan)
    print('Frozen',len(cells),'roots;',sum(not c['dendritic_source_available'] for c in cells),'without dendritic source',flush=True)

def prepare(rank,tasks,workers):
    import torch
    from data.dataset_presynaptic import load_cell_arrays
    from scripts.subcompartment_predict_all import load_models,HIERARCHY
    from scripts.subcompartment_predict_fast import fold_predictor
    torch.set_num_threads(1);plan=frozen_load();digest=sha(BASE/'plan.json')
    models=load_models(plan,torch.device('cpu'));predict=fold_predictor(models,[m['classes'] for m in plan['models']])
    prediction_lock=threading.Lock()
    with torch.inference_mode():predict(torch.zeros((1,64)))
    labels=HIERARCHY.level_classes[0]
    nuclei=json.loads((ROOT/'data/nucleus_positions.json').read_text())['positions']
    def classify(x,mask):
        means=x.sum(1)/mask.sum(1)[:,None];outputs=[]
        with torch.inference_mode():
            for start in range(0,len(means),8192):
                # functional_call temporarily replaces template parameters: protect
                # it while independent root reads and geometry decoding stay parallel.
                with prediction_lock:
                    logits=predict(torch.from_numpy(means[start:start+8192].astype(np.float32)))
                outputs.append(logits.argmax(-1).numpy().T.astype(np.int8))
        return np.concatenate(outputs) if outputs else np.empty((0,5),np.int8)
    def one(info):
        r=info['root_id'];dest=BASE/'cells'/r
        if (dest/'complete.json').exists():
            assert json.loads((dest/'complete.json').read_text())['plan_sha256']==digest
            return
        temp=dest.with_name(f'.{r}.{os.getpid()}');temp.mkdir(parents=True,exist_ok=True)
        cell=load_cell_arrays(Path(info['source']),'cave')
        ids=cell['observed_ids'];lookup=np.full(len(cell['pos']),-1,np.int64);lookup[ids]=np.arange(len(ids))
        with np.load(info['source']) as z,np.load(POST/'cells'/f'{r}.npz') as post:
            np.testing.assert_array_equal(z['cave_synapse_id'],post['synapse_ids'])
            rows=np.flatnonzero(z['cave_valid_k_window'] & post['eligible']);kept=[];seen=set()
            for row in rows:
                a,b=cell['offsets'][row:row+2];nodes=cell['members'][a:b]
                key=np.sort(nodes).tobytes()
                if key not in seen:seen.add(key);kept.append(row)
            rows=np.asarray(kept,np.int64)
            mapped=lookup[cell['members']]
            if (mapped<0).any():raise ValueError('Missing CAVE embeddings')
            x,mask=padded(cell['observed_x'],cell['offsets'],mapped,rows)
            aux=post['post_x'][rows].astype(np.float32);syn=post['synapse_ids'][rows].astype(np.int64)
            if not np.isfinite(aux).all():raise ValueError('Nonfinite post')
        comp=classify(x,mask)
        for name,value in dict(axon_x=x,axon_mask=mask,axon_aux=aux,axon_comp=comp,
                               axon_synapse_id=syn,axon_source_row=rows).items():np.save(temp/f'{name}.npy',value)
        counts=dict(axon_total=len(x),axon_by_fold=[int((comp[:,f]==labels.index('axon')).sum()) for f in range(5)])
        if info['dendritic_source_available']:
            graph=torch.load(ROOT/'data/graph_cache'/f'{r}.pt',map_location='cpu',weights_only=False)
            with np.load(NEIGHBOR/f'{r}.npz') as z:
                cache_index=z['cache_index'];offsets=z['offsets'];members=cache_index[z['members']]
                if r not in nuclei:raise ValueError('Dendritic source missing nucleus')
                distances=np.linalg.norm(graph.pos.numpy()[cache_index]-np.asarray(nuclei[r]),axis=1)
                assert (distances>=5000).all()
                # One window per unique node set, matching the presynaptic loader's unit.
                rows=[];seen=set()
                for row in range(len(offsets)-1):
                    a,b=offsets[row:row+2];key=np.sort(members[a:b]).tobytes()
                    if key not in seen:seen.add(key);rows.append(row)
                rows=np.asarray(rows,np.int64)
                x,mask=padded(graph.x.numpy(),offsets,members,rows)
                centers=cache_index[rows]
            aux=(x.sum(1)/mask.sum(1)[:,None]).astype(np.float32);comp=classify(x,mask)
        else:
            x=np.empty((0,10,64),np.float32);mask=np.empty((0,10),bool)
            aux=np.empty((0,64),np.float32);comp=np.empty((0,5),np.int8);centers=np.empty(0,np.int64)
        for name,value in dict(dendrite_x=x,dendrite_mask=mask,dendrite_aux=aux,dendrite_comp=comp,
                               dendrite_center=centers).items():np.save(temp/f'{name}.npy',value)
        counts.update(dendrite_total=len(x),dendrite_by_fold=[int((comp[:,f]==labels.index('dendrite')).sum()) for f in range(5)])
        atomic_json(temp/'complete.json',dict(root_id=r,plan_sha256=digest,**counts))
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():shutil.rmtree(dest)
        temp.replace(dest);print(json.dumps(dict(root_id=r,**counts)),flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(one,plan['cells'][rank::tasks]))

def split_roots(manifest,plan,fold):
    """Stratified inner holdout uses outer-training cells only."""
    train=[];test=[];strata={i:[] for i in range(6)}
    for i,info in enumerate(plan['cells']):
        record=json.loads((BASE/'cells'/info['root_id']/'complete.json').read_text())
        if not record['axon_by_fold'][fold]:continue
        split=manifest['cells'][info['root_id']]['split']
        if split=='test':test.append(i)
        elif split=='train':strata[info['targets'][-1]].append(i)
        else:raise ValueError(split)
    rng=np.random.default_rng(1000+fold);validation=[]
    for label,roots in strata.items():
        if len(roots)<2:raise ValueError(f'Not enough train roots for class {label}')
        roots=np.asarray(roots);rng.shuffle(roots);n=min(len(roots)-1,max(1,round(len(roots)*.1)))
        validation.extend(roots[:n].tolist());train.extend(roots[n:].tolist())
    assert not(set(train)&set(test) or set(train)&set(validation) or set(test)&set(validation))
    return dict(train=sorted(train),validation=sorted(validation),test=sorted(test))

def finalize():
    from scripts.subcompartment_predict_all import HIERARCHY
    plan=frozen_load();digest=sha(BASE/'plan.json');labels=HIERARCHY.level_classes[0]
    for info in plan['cells']:
        z=json.loads((BASE/'cells'/info['root_id']/'complete.json').read_text());assert z['plan_sha256']==digest
    summary=[]
    for fold in range(5):
        manifest=json.loads(Path(plan['manifest_paths'][fold]).read_text());splits=split_roots(manifest,plan,fold)
        for split,root_indices in splits.items():
            kinds=['axon','dendrite'] if split=='train' else ['axon']
            for condition in (['axon_only','axon_dendrite'] if split=='train' else ['shared']):
                records=[]
                for i in root_indices:
                    for kind in kinds:
                        if condition=='axon_only' and kind=='dendrite':continue
                        comp=np.load(BASE/'cells'/plan['cells'][i]['root_id']/f'{kind}_comp.npy',mmap_mode='r')
                        selected=np.flatnonzero(comp[:,fold]==labels.index('axon' if kind=='axon' else 'dendrite'))
                        records.append(np.column_stack([np.full(len(selected),i),np.full(len(selected),kind=='dendrite'),selected]).astype(np.int32))
                record=np.concatenate(records) if records else np.empty((0,3),np.int32)
                path=BASE/'indices'/f'fold{fold}_{split}_{condition}.npy';path.parent.mkdir(exist_ok=True)
                np.save(path,record)
                summary.append(dict(fold=fold,split=split,condition=condition,roots=len(root_indices),windows=len(record)))
        atomic_json(BASE/'indices'/f'fold{fold}_splits.json',{k:[plan['cells'][i]['root_id'] for i in v] for k,v in splits.items()})
        a=np.load(BASE/'indices'/f'fold{fold}_train_axon_only.npy',mmap_mode='r')
        b=np.load(BASE/'indices'/f'fold{fold}_train_axon_dendrite.npy',mmap_mode='r')
        np.testing.assert_array_equal(a,b[b[:,1]==0])
        assert len(a)>0
    atomic_json(BASE/'complete.json',dict(plan_sha256=digest,summary=summary,shared_test_windows=True,
                                       dendritic_source_missing_roots=[c['root_id'] for c in plan['cells'] if not c['dendritic_source_available']]))
    atomic_json(DOC/'preparation_summary.json',json.loads((BASE/'complete.json').read_text()))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','prepare','finalize']);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--tasks',type=int,default=32);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.stage=='initialize':initialize()
    elif a.stage=='prepare':prepare(a.rank,a.tasks,a.workers)
    else:finalize()
