"""Five-fold pointwise compartment predictions, with immutable cache and DB import.

The 10-um exclusion describes model training, not an inference mask. Every
available node is classified; no averaging or geodesic pooling is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from segclr_db import store as st
from segclr_db.identity import LabelHierarchy, PredictionRun

DB_ROOT = '/orcd/compute/sdorkenw/001/segclr-db'
OUT = Path('/orcd/scratch/orcd/013/jcbliao/subcompartment_geodesic10um_predictions_v1')
ROUTES = Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911/routes.json')
CV = REPO / 'results/subcompartment_geodesic_10um_cv'
CAVE_H5 = Path('/orcd/compute/sdorkenw/001/collina/data/all_cells_v2_embs_1718_resnet_860b_reshuffled_20260603_150412_checkpoint_e0_s95000.h5')
EMBED_RUN = 'resnet_860b_reshuffled__20260603_150412'
EMBED_CKPT = 'checkpoint_e0_s95000'
NEURONS = {'L2IT','L3IT','L4IT','L5IT','L6IT','L5ET','L5NP','L6CT',
           'PV','MC','NMC','ITC','ITCperi','DTC','NGC','L1','AltBasket','AltDTC','ChC','thalamocortical'}
HIERARCHY = LabelHierarchy.from_tree('subcompartment_four_class_v1',
    {name: None for name in ['soma','dendrite','axon','astrocytic_process']})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **values)
    tmp.replace(path)


def db_payload(logits, classes):
    """Reorder raw model logits into the hierarchy's explicit column order."""
    values = np.asarray(logits, np.float32)
    if len(set(classes)) != len(classes) or set(classes) != set(HIERARCHY.level_classes[0]):
        raise ValueError('checkpoint classes do not match the compartment hierarchy')
    if values.ndim != 2 or values.shape[1] != len(classes) or not np.isfinite(values).all():
        raise ValueError('expected finite N x 4 raw logits')
    ordered = values[:, [classes.index(c) for c in HIERARCHY.level_classes[0]]]
    labels = np.asarray(HIERARCHY.level_classes[0], object)[ordered.argmax(1)][:, None]
    shifted = ordered - ordered.max(1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(1, keepdims=True)
    # Predictive entropy, in nats; this is not an SNGP uncertainty estimate.
    uncertainty = -(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(1)[:, None]
    return labels, uncertainty.astype(np.float32), [ordered]


def unique_embedding_rows(nodes, vectors):
    """Remove duplicated HDF5 export rows, refusing differing node values.

    Repeated cell-label rows caused five astrocytes to be exported twice.
    Copies differ by at most float32 rounding; retain the first, never pool.
    """
    order=np.argsort(nodes,kind='stable');nodes=np.asarray(nodes)[order];vectors=np.asarray(vectors)[order]
    duplicate=nodes[1:]==nodes[:-1]
    if duplicate.any():
        np.testing.assert_allclose(vectors[1:][duplicate],vectors[:-1][duplicate],rtol=1e-6,atol=1e-6)
    keep=np.r_[True,~duplicate] if len(nodes) else np.empty(0,bool)
    return nodes[keep],vectors[keep]


def prepare(out, tasks):
    import torch
    store = st.open_store(DB_ROOT, 'microns')
    labels = st.scan(store, 'cell_labels', filter="label_set = 'cell_type'").to_pylist()
    labels = sorted([r for r in labels if r['label'] in NEURONS | {'astrocyte'}], key=lambda r:r['root_id'])
    unique = {}
    for row in labels:
        old = unique.setdefault(row['root_id'], row)
        if old['label'] != row['label']: raise ValueError('ambiguous cell labels')
    labels = list(unique.values())
    roots = [r['root_id'] for r in labels]
    routes = {r['root_id']:r for r in json.loads(ROUTES.read_text())}
    named = {(r['root_id'],r['skeleton_name']):r['n_nodes'] for r in st.scan(
        store, 'named_skeletons', columns=['root_id','skeleton_name','n_nodes'], root_ids=roots).to_pylist()}
    cave = {r['root_id']:r['n_nodes'] for r in st.scan(store, 'skeleton_manifest',
        columns=['root_id','n_nodes'], root_ids=roots).to_pylist()}
    fold_plan = json.loads((CV/'folds.json').read_text())
    models = []
    for fold in range(5):
        path = Path(fold_plan['fold0_model']) if fold == 0 else CV/f'fold{fold}/models_resnet_inverse_sqrt_100epochs/geodesic/model.pt'
        ck = torch.load(path, map_location='cpu', weights_only=True)
        assert ck['variant']=='geodesic' and ck['head']=='resnet' and ck['embedding_dim']==64
        assert ck.get('fold', fold)==fold
        db_payload(np.zeros((1,4)), ck['classes'])
        models.append(dict(fold=fold, path=str(path), sha256=sha(path), classes=ck['classes'], best_epoch=ck['best_epoch']))
    import h5py
    with h5py.File(CAVE_H5) as exported:
        exported_roots = exported['seg_ids'][:]
        boundaries = np.r_[0, np.flatnonzero(exported_roots[1:] != exported_roots[:-1])+1, len(exported_roots)]
        h5_ranges = {}
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            h5_ranges.setdefault(int(exported_roots[start]), []).append([int(start),int(stop)])
    h5_stat=CAVE_H5.stat()
    items = []; missing = []
    for cell in labels:
        root = cell['root_id']
        if root in cave:
            items.append(dict(root_id=root, geometry='cave', skeleton_name=None, n_nodes=cave[root],
                h5_ranges=h5_ranges.get(root,[])))
        else:
            missing.append(dict(root_id=root, geometry='cave', reason='no cached geometry'))
        route = routes.get(root)
        if route:
            dense_name = route['skeleton_name']
            raw_name = dense_name.replace('_111nm', '')
            if (root, raw_name) in named:
                items.append(dict(root_id=root, geometry='teasar', skeleton_name=raw_name,
                    embedding_skeleton_name=dense_name, n_nodes=named[root,raw_name]))
            else:
                missing.append(dict(root_id=root, geometry='teasar', reason=f'no original geometry {raw_name}'))
        else:
            missing.append(dict(root_id=root, geometry='teasar', reason='no original TEASAR route'))
    # Existing neuronal cache is frozen and already being embedded independently.
    presyn = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_point_embeddings')
    point_plan = json.loads((presyn/'plan.json').read_text())
    for root in sorted({c['root_id'] for c in point_plan['chunks']} & set(roots)):
        chunks = sorted([c for c in point_plan['chunks'] if c['root_id']==root], key=lambda c:c['start'])
        items.append(dict(root_id=root, geometry='presynaptic', skeleton_name='presynaptic_sites',
            point_cache=str(presyn), chunks=chunks, n_nodes=sum(c['count'] for c in chunks)))
    # Contiguous root blocks aid Lance fragment pruning and avoid duplicate ownership.
    items.sort(key=lambda x:(x['root_id'],x['geometry']))
    for rank, block in enumerate(np.array_split(np.arange(len(items)), tasks)):
        for i in block: items[int(i)]['task_id']=rank
    tables = {}
    for table in ['node_embeddings','named_node_embeddings','named_skeletons']:
        ds = st.open_table(store, table, 64 if 'embeddings' in table else None)
        tables[table] = dict(uri=ds.uri, version=ds.version)
    plan = dict(models=models, fold_plan=str(CV/'folds.json'), fold_plan_sha256=sha(CV/'folds.json'),
        embedding_run_id=EMBED_RUN, embedding_checkpoint_id=EMBED_CKPT, agg_spec_id=None, window_nm=None,
        hierarchy_id=HIERARCHY.hierarchy_id, logit_classes=list(HIERARCHY.level_classes[0]),
        uncertainty='categorical entropy in nats', mat_version=1718, num_tasks=tasks,
        cells=[dict(root_id=r['root_id'],cell_type=r['label']) for r in labels], items=items,
        missing_geometry=missing, tables=tables, presynaptic_cohort='neurons only',
        cave_h5=dict(path=str(CAVE_H5),size=h5_stat.st_size,mtime_ns=h5_stat.st_mtime_ns),
        training_soma_exclusion_nm=10000, inference_soma_mask=False,
        original_teasar_only=True)
    atomic_json(out/'plan.json', plan)
    print(json.dumps(dict(cells=len(labels), items=len(items), nodes=sum(i['n_nodes'] for i in items),
        missing_geometry=missing), indent=2), flush=True)


def load_models(plan, device):
    import torch
    sys.path.insert(0, str(REPO.parent/'subcompartment_classification'))
    from train_compartment_gpu import make_model
    result = []
    for info in plan['models']:
        assert sha(info['path']) == info['sha256']
        ck = torch.load(info['path'], map_location='cpu', weights_only=True)
        model = make_model(ck['head'], ck['state_dict']['mean'], ck['state_dict']['scale'])
        model.load_state_dict(ck['state_dict'], strict=True)
        result.append(model.to(device).eval())
    return result


def destination(out, item):
    return out/'predictions'/item['geometry']/str(item['root_id'])/'all_folds.npz'


def read_points(item):
    nodes=[]; vectors=[]; coords=[]; ids=[]
    for chunk in item['chunks']:
        path = Path(item['point_cache'])/'parts'/str(item['root_id'])/f"{chunk['start']:08d}.npz"
        with np.load(path) as z:
            assert int(z['root_id'])==item['root_id'] and len(z['embeddings'])==chunk['count']
            nodes.append(np.arange(chunk['start'],chunk['start']+chunk['count'],dtype=np.int32))
            vectors.append(z['embeddings']); coords.append(z['positions_nm']); ids.append(z['synapse_ids'])
    return np.concatenate(nodes), np.concatenate(vectors), np.concatenate(coords), np.concatenate(ids)


def infer(out, rank, pilot=False):
    import torch
    import lance
    from segclr_db.skeletons import SkeletonCache
    plan = json.loads((out/'plan.json').read_text()); plan_hash = sha(out/'plan.json')
    if not torch.cuda.is_available() and not pilot: raise RuntimeError('GPU required for production inference')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    models = load_models(plan,device)
    store = st.open_store(DB_ROOT,'microns'); cache = SkeletonCache(store)
    tables = {k:lance.dataset(v['uri'],version=v['version']) for k,v in plan['tables'].items()}
    items = [i for i in plan['items'] if i['task_id']==rank]
    if pilot:
        items = [min([i for i in plan['items'] if i['geometry']==g and (g!='presynaptic' or all(
                    (Path(i['point_cache'])/'parts'/str(i['root_id'])/f"{c['start']:08d}.npz").exists() for c in i['chunks']))],key=lambda x:x['n_nodes'])
                 for g in ['cave','teasar','presynaptic']]
    failures=[]
    engine=None
    for item in items:
        dest=destination(out,item)
        try:
            if dest.exists() and not pilot:
                with np.load(dest) as z:
                    assert str(z['plan_sha256'])==plan_hash
                    assert z['logits'].shape==(5,item['n_nodes'],4)
                    np.testing.assert_array_equal(z['node_ids'],np.arange(item['n_nodes']))
                continue
            root=item['root_id']; n=item['n_nodes']; coords=ids=None
            if item['geometry']=='presynaptic':
                nodes, x, coords, ids=read_points(item)
            else:
                table='node_embeddings' if item['geometry']=='cave' else 'named_node_embeddings'
                clause=f"root_id = {root} AND run_id = '{EMBED_RUN}' AND checkpoint_id = '{EMBED_CKPT}' AND node_id < {n}"
                if item['geometry']=='teasar':
                    name=item['embedding_skeleton_name']; clause+=f" AND skeleton_name = '{name}'"
                    original=cache.get_named_skeleton(root,item['skeleton_name'])
                    # Match by the preserved node identity and exact position, never nearest-neighbour.
                    dense=tables['named_skeletons'].to_table(columns=['coords'],
                        filter=f"root_id = {root} AND skeleton_name = '{name}'").to_pylist()
                    if len(dense)!=1: raise ValueError('missing embedding-source geometry')
                    np.testing.assert_array_equal(original.coords,np.asarray(dense[0]['coords'],np.float32)[:n])
                if item['geometry']=='cave' and item['h5_ranges']:
                    import h5py
                    info=plan['cave_h5']; h5_path=Path(info['path']);stat=h5_path.stat()
                    assert stat.st_size==info['size'] and stat.st_mtime_ns==info['mtime_ns']
                    with h5py.File(h5_path) as exported:
                        nodes=np.concatenate([exported['nodes'][a:b] for a,b in item['h5_ranges']]).astype(np.int32)
                        x=np.concatenate([exported['embeddings'][a:b] for a,b in item['h5_ranges']])
                    nodes,x=unique_embedding_rows(nodes,x)
                else:
                    try:
                        rows=tables[table].to_table(columns=['node_id','embedding'],filter=clause).sort_by('node_id')
                        nodes=rows['node_id'].to_numpy().astype(np.int32)
                        x=np.asarray(rows['embedding'].combine_chunks().values).reshape(-1,64)
                    except Exception as error:
                        if item['geometry']!='cave' or 'Permission denied' not in str(error): raise
                        # Newly included cells absent from the published export need fresh crops.
                        nodes=np.empty(0,np.int32);x=np.empty((0,64),np.float32)
                if len(np.unique(nodes))!=len(nodes): raise ValueError('duplicate source embeddings')
                if len(nodes)!=n:
                    # Fill gaps at this geometry's own positions using the same pinned SegCLR weights.
                    if pilot: raise ValueError(f'missing {n-len(nodes)} embeddings; pilot requires complete source')
                    if engine is None:
                        from scripts.generate_cave_embedding_augmentations import FastAugInference
                        engine=FastAugInference('clean',0,0,batch_size=32,num_threads=8)
                    geometry=(cache.get_skeleton(root,fetch_if_missing=False) if item['geometry']=='cave' else original)
                    missing=np.setdiff1d(np.arange(n,dtype=np.int32),nodes)
                    torch.backends.cuda.matmul.allow_tf32=True
                    got,new=engine.embed(root,geometry.coords,missing)
                    np.testing.assert_array_equal(got,missing)
                    save_npz(out/'embedding_backfill'/item['geometry']/f'{root}.npz',
                        node_ids=got,embeddings=new,root_id=root,skeleton_name=item['skeleton_name'] or '',
                        run_id=EMBED_RUN,checkpoint_id=EMBED_CKPT)
                    nodes=np.concatenate([nodes,got]);x=np.concatenate([x,new]);order=np.argsort(nodes);nodes=nodes[order];x=x[order]
            np.testing.assert_array_equal(nodes,np.arange(n))
            if x.shape!=(n,64) or not np.isfinite(x).all(): raise ValueError('invalid embedding coverage')
            if pilot: x=x[:min(n,128)]; nodes=nodes[:len(x)]
            logits=np.empty((5,len(x),4),np.float32)
            torch.backends.cuda.matmul.allow_tf32=False
            with torch.inference_mode():
                for start in range(0,len(x),65536):
                    features=torch.from_numpy(np.array(x[start:start+65536],dtype=np.float32)).to(device)
                    for fold,model in enumerate(models):
                        raw=model(features).cpu().numpy()
                        logits[fold,start:start+len(features)]=db_payload(raw,plan['models'][fold]['classes'])[2][0]
            if not pilot:
                values=dict(node_ids=nodes,logits=logits,root_id=root,skeleton_name=item['skeleton_name'] or '',
                    classes=np.asarray(plan['logit_classes']),plan_sha256=plan_hash)
                if coords is not None: values.update(positions_nm=coords,synapse_ids=ids)
                save_npz(dest,**values)
            print(json.dumps(dict(root_id=root,geometry=item['geometry'],nodes=len(nodes),folds=5,pilot=pilot)),flush=True)
        except Exception as exc:
            failures.append(dict(item=item,error=repr(exc)))
            print(json.dumps(failures[-1]),flush=True)
    atomic_json(out/'status'/f"{'pilot' if pilot else 'infer'}_{rank}.json",dict(failures=failures,items=len(items)))
    if failures: raise RuntimeError(f'{len(failures)} cells failed; see status report')


def register(writer, plan):
    import pandas as pd
    writer.register_hierarchy(HIERARCHY)
    experiment='subcompartment_resnet_geodesic_soma_exclusion10um_cv5'
    writer.register_experiment(experiment,dict(head='resnet',hidden_size=128,hidden_layers=4,
        embedding_dim=64,soma_exclusion_nm=10000,variant='geodesic',sampling='inverse_sqrt',
        fold_plan_sha256=plan['fold_plan_sha256'],agg_spec_id=None),kind='classifier',hierarchy_id=HIERARCHY.hierarchy_id)
    assignments=json.loads(Path(plan['fold_plan']).read_text())['assignments']
    ids=[]
    for info in plan['models']:
        fold=info['fold'];path=Path(info['path']);assert sha(path)==info['sha256']
        run=writer.register_run(experiment,path.parent,f'20261008_00000{fold}',seed=0)
        checkpoint=f"best_epoch{info['best_epoch']}_{info['sha256'][:12]}"
        writer.register_checkpoints(run.run_id,[(checkpoint,info['best_epoch'],0,str(path))])
        split_id=f'subcompartment_geodesic10um_cv5_fold{fold}'
        frame=pd.DataFrame([dict(root_id=int(root),split='test' if held==fold else 'train') for root,held in assignments.items()])
        writer.create_split(split_id,frame,source_key=plan['fold_plan'],seed=0,fold=fold)
        writer.link_run_split(run.run_id,split_id)
        ids.append(writer.register_prediction_run(PredictionRun(classifier_run_id=run.run_id,
            classifier_checkpoint_id=checkpoint,embedding_run_id=EMBED_RUN,embedding_checkpoint_id=EMBED_CKPT,
            agg_spec_id=None,window_nm=None,notes=json.dumps(dict(model_sha256=info['sha256'],
                fold=fold,uncertainty=plan['uncertainty'],inference='pointwise, no aggregation')))))
    return ids


def import_predictions(out):
    from segclr_db.writer import SegCLRWriter
    from segclr_db.experiment import SegCLRExperiment
    plan=json.loads((out/'plan.json').read_text());plan_hash=sha(out/'plan.json')
    store=st.open_store(DB_ROOT,'microns')
    # Fail before any mutations if the provenance registry is inaccessible.
    st.scan(store,'runs',columns=['run_id'])
    writer=SegCLRWriter(store=store);experiment=SegCLRExperiment.from_registry('resnet_860b_reshuffled',store)
    runs=register(writer,plan);atomic_json(out/'prediction_runs.json',runs)
    missing=[];written=0
    for item in plan['items']:
        dest=destination(out,item)
        if not dest.exists(): missing.append(item);continue
        with np.load(dest) as z:
            assert str(z['plan_sha256'])==plan_hash
            assert z['logits'].shape==(5,item['n_nodes'],4)
            root=item['root_id'];name=item['skeleton_name']
            if item['geometry']=='presynaptic':
                # Also imports neuronal point geometry if its independent import is pending.
                _,vectors,coords,ids=read_points(item)
                writer.add_named_point_embeddings(experiment,root,name,coords,vectors,ids,checkpoint_id=EMBED_CKPT)
            for fold,run in enumerate(runs):
                labels,uncertainty,logits=db_payload(z['logits'][fold],list(z['classes']))
                written+=writer.add_cell_predictions(run,root,z['node_ids'],labels,uncertainty,logits,
                    synapse_ids=z['synapse_ids'] if item['geometry']=='presynaptic' else None,skeleton_name=name)
    atomic_json(out/'import_report.json',dict(written=written,missing=missing,prediction_runs=runs))
    if missing: raise RuntimeError(f'{len(missing)} prediction files missing')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','infer','pilot','import']);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--tasks',type=int,default=16);p.add_argument('--rank',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':
        if (a.output/'plan.json').exists(): raise ValueError('plan already exists; use frozen plan')
        prepare(a.output,a.tasks)
    elif a.mode=='import':import_predictions(a.output)
    else:infer(a.output,a.rank,pilot=a.mode=='pilot')

if __name__=='__main__':main()
