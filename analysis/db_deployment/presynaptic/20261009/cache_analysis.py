"""Build reproducible local analysis caches; never writes shared segclr_db."""
from pathlib import Path
import argparse, hashlib, json, os, sys
import numpy as np
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'scripts/train_gnn.py').exists())
sys.path.insert(0,str(ROOT))
SOURCE = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2/cell_types')
CACHE = Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/db_deployment/presynaptic/20261009')
BINS = np.linspace(0,1,51)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.stem}.{os.getpid()}.npz')
    np.savez_compressed(tmp,**arrays);os.replace(tmp,path)

def build_test(family,fold,workers):
    import torch
    from torch_geometric.loader import DataLoader
    from data.dataset_presynaptic import PresynapticWindowDataset
    from scripts.predict_incoming_fragment_cell_types import checkpoint,load_checkpoint
    torch.set_num_threads(max(1,workers))
    path=checkpoint(family,'six',fold); model,manifest=load_checkpoint(path)
    args=json.loads(path.parent.with_suffix('.json').read_text())['args']
    provenance=dict(checkpoint_sha256=digest(path),manifest_sha256=digest(manifest),args=args,format=1)
    dest=CACHE/family/f'test_fold{fold}.npz'
    if dest.exists():
        with np.load(dest) as z:
            if json.loads(str(z['provenance']))==provenance:return
    ds=PresynapticWindowDataset(json.loads(manifest.read_text()),'test','cave' if args['dataset']=='presynaptic_cave' else 'new',
        database=args['presynaptic_database'],cell_cache_size=4,
        postsynaptic_cache=args['postsynaptic_cache'],use_postsynaptic=True,
        single_presynaptic_embedding=args.get('single_presynaptic_embedding',False),
        memmap_root=args.get('presynaptic_memmap_root'),compartment_filter=args.get('presynaptic_compartment_filter'))
    assert ds.classes==model.cls_head.hierarchy.level_classes[-1]
    loader=DataLoader(ds,batch_size=2048,shuffle=False,num_workers=workers,persistent_workers=workers>0)
    values={k:[] for k in ('probabilities','prediction','target','root_id','synapse_id')}
    with torch.inference_mode():
        for batch in loader:
            hidden=model(batch.x,batch.edge_index,batch.batch,pos_enc=batch.pos_enc,rel_pos=batch.rel_pos,
                         has_segclr=batch.has_segclr,postsynaptic_embedding=batch.postsynaptic_embedding)
            decoded=model.cls_head.predict_distribution(hidden)
            values['probabilities'].append(decoded['level_probabilities'][-1].numpy())
            values['prediction'].append(decoded['predictions'][:,-1].numpy())
            values['target'].append(batch.y_levels[:,-1].numpy())
            values['root_id'].append(batch.root_id.numpy().reshape(-1))
            values['synapse_id'].append(batch.synapse_id.numpy().reshape(-1))
    arrays={k:np.concatenate(v) for k,v in values.items()}
    assert len(arrays['target'])==len(ds)
    assert np.allclose(arrays['probabilities'].sum(1),1,atol=1e-5)
    metrics=json.loads(path.parent.with_suffix('.json').read_text())['window_test_metrics']
    accuracy=float(np.mean(arrays['prediction']==arrays['target']))
    print('FP32 vs saved AMP accuracy',accuracy,metrics['accuracy'],flush=True)
    assert abs(accuracy-metrics['accuracy'])<0.002, 'Reconstructed test accuracy differs materially from saved evaluation'
    save(dest,**arrays,classes=np.asarray(ds.classes),provenance=json.dumps(provenance,sort_keys=True))
    print('test',family,fold,len(ds),flush=True)

def build_unlabeled(family,fold,workers):
    import duckdb
    plan=json.loads((SOURCE/'plan.json').read_text())
    info=next(m for m in plan['models'] if m['family']==family and m['classes']=='six' and m['fold']==fold)
    provenance=dict(plan_sha256=digest(SOURCE/'plan.json'),format=2,family=family,fold=fold)
    dest=CACHE/family/f'unlabeled_fold{fold}.npz'
    if dest.exists():
        with np.load(dest) as z:
            if json.loads(str(z['provenance']))==provenance:return
    con=duckdb.connect();con.execute(f'SET threads={max(1,workers)}');con.execute("SET memory_limit='8GB'")
    prefix=f'{family}_six_fold{fold}';p=prefix+'_level2_probabilities';pred=prefix+'_level2_class_index'
    source=str(SOURCE/'parts/*/[0-9][0-9][0-9][0-9][0-9].parquet')
    query=f'''SELECT {prefix}_n_embeddings_used AS nodes, {p}[{pred}+1] AS selected,
              {','.join(f'{p}[{i+1}] AS p{i}' for i in range(6))}, {pred} AS prediction
              FROM read_parquet('{source}') WHERE {prefix}_included'''
    hist=np.zeros(50,dtype=np.int64);n=np.zeros(10,dtype=np.int64)
    sums=np.zeros((10,7));squares=np.zeros((10,7))
    class_n=np.zeros((10,6),dtype=np.int64);class_sums=np.zeros((10,6));class_squares=np.zeros((10,6))
    for batch in con.execute(query).fetch_record_batch(131072):
        nodes=batch.column(0).to_numpy().astype(int)-1
        scores=np.column_stack([batch.column(i).to_numpy() for i in range(1,8)])
        assert np.isfinite(scores).all() and ((nodes>=0)&(nodes<10)).all()
        prediction=batch.column(8).to_numpy().astype(int)
        for j in range(6):
            mask=prediction==j
            class_n[:,j]+=np.bincount(nodes[mask],minlength=10)
            class_sums[:,j]+=np.bincount(nodes[mask],weights=scores[mask,0],minlength=10)
            class_squares[:,j]+=np.bincount(nodes[mask],weights=scores[mask,0]**2,minlength=10)
        hist+=np.histogram(scores[:,0],BINS)[0]
        n+=np.bincount(nodes,minlength=10)
        for j in range(7):
            sums[:,j]+=np.bincount(nodes,weights=scores[:,j],minlength=10)
            squares[:,j]+=np.bincount(nodes,weights=scores[:,j]**2,minlength=10)
    assert np.array_equal(class_n.sum(1),n)
    assert np.allclose(class_sums.sum(1),sums[:,0])
    save(dest,hist=hist,bins=BINS,n=n,sums=sums,squares=squares,
         selected_class_n=class_n,selected_class_sums=class_sums,selected_class_squares=class_squares,
         classes=np.asarray(info['level_classes'][-1]),provenance=json.dumps(provenance,sort_keys=True))
    print('unlabeled',family,fold,int(n.sum()),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True,choices=range(5))
    ap.add_argument('--family',choices=['cave_n10','single_pre_post'],default='cave_n10')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--stage',choices=['test','unlabeled','both'],default='both')
    a=ap.parse_args()
    if a.stage in ('test','both'):build_test(a.family,a.fold,a.workers)
    if a.stage in ('unlabeled','both'):build_unlabeled(a.family,a.fold,a.workers)
