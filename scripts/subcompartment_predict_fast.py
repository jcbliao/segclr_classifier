"""Batched reads, prefetched inputs, vectorized fold inference, async result writes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.subcompartment_predict_all import (
    OUT, EMBED_RUN, EMBED_CKPT, HIERARCHY, load_models, destination, read_points,
    unique_embedding_rows, sha, save_npz, atomic_json,
)


def chunks(items, max_cells=12, max_nodes=500_000):
    batch=[];nodes=0
    for item in items:
        if batch and (len(batch)>=max_cells or nodes+item['n_nodes']>max_nodes):
            yield batch;batch=[];nodes=0
        batch.append(item);nodes+=item['n_nodes']
    if batch:yield batch


def pending_items(plan,out,rank):
    # Only this shard owns these paths. Atomic renames preserve already completed files.
    return [i for i in plan['items'] if i['task_id']==rank and not destination(out,i).exists()]


def coords_from_arrow(value):
    # List<FixedSizeList<float,3>> -> contiguous Arrow values, without Python objects.
    return value.values.flatten().to_numpy(zero_copy_only=False).reshape(-1,3)


def check_geometry(table, items):
    roots=table['root_id'].to_numpy();names=table['skeleton_name'].to_pylist()
    index={(int(root),name):i for i,(root,name) in enumerate(zip(roots,names))}
    for item in items:
        root=item['root_id'];n=item['n_nodes']
        raw=coords_from_arrow(table['coords'][index[root,item['skeleton_name']]])
        dense=coords_from_arrow(table['coords'][index[root,item['embedding_skeleton_name']]])
        if len(raw)!=n:raise ValueError(f'original geometry size changed for {root}')
        np.testing.assert_array_equal(raw,dense[:n])


def partition_embeddings(table, items, allow_missing=False):
    roots=table['root_id'].to_numpy();nodes=table['node_id'].to_numpy()
    x=table['embedding'].combine_chunks().values.to_numpy().reshape(-1,64)
    order=np.lexsort((nodes,roots));roots=roots[order];nodes=nodes[order];x=x[order]
    result={}
    for item in items:
        root=item['root_id'];a,b=np.searchsorted(roots,root,side='left'),np.searchsorted(roots,root,side='right')
        node=nodes[a:b];vec=x[a:b]
        if allow_missing:
            if len(node) and (node[0]<0 or node[-1]>=item['n_nodes'] or np.any(node[1:]<=node[:-1])):
                raise ValueError(f'invalid or duplicate source node IDs for {root}')
        else:
            np.testing.assert_array_equal(node,np.arange(item['n_nodes']))
        if not np.isfinite(vec).all():raise ValueError(f'nonfinite embeddings for {root}')
        result[root]=(node,vec)
    return result


def merge_backfill(nodes,x,new_nodes,new_x,n):
    nodes=np.asarray(nodes,np.int32);new_nodes=np.asarray(new_nodes,np.int32)
    new_x=np.asarray(new_x,np.float32)
    if new_x.shape!=(len(new_nodes),64) or not np.isfinite(new_x).all():
        raise ValueError('invalid cached backfill vectors')
    combined=np.concatenate([nodes,new_nodes]);vectors=np.concatenate([x,new_x])
    order=np.argsort(combined);combined=combined[order];vectors=vectors[order]
    if len(combined) and (combined[0]<0 or combined[-1]>=n or np.any(combined[1:]<=combined[:-1])):
        raise ValueError('backfill overlaps source or contains invalid node IDs')
    return combined,vectors


def complete_embeddings(loaded,out,plan_hash,get_engine):
    """Fill only absent original nodes, on the GPU thread rather than loader threads."""
    import torch
    result=[]
    for item,nodes,x,extra in loaded:
        missing=extra.pop('_missing_node_ids',None)
        coords=extra.pop('_original_coords',None)
        if missing is not None:
            engine=get_engine();torch.backends.cuda.matmul.allow_tf32=True
            got,new=engine.embed(item['root_id'],coords,missing)
            np.testing.assert_array_equal(got,missing)
            nodes,x=merge_backfill(nodes,x,got,new,item['n_nodes'])
            save_npz(out/'embedding_backfill'/'teasar'/f"{item['root_id']}.npz",
                root_id=item['root_id'],skeleton_name=item['skeleton_name'],node_ids=got,embeddings=new,
                run_id=EMBED_RUN,checkpoint_id=EMBED_CKPT,plan_sha256=plan_hash)
            print(json.dumps(dict(stage='backfill',root_id=item['root_id'],nodes=len(got))),flush=True)
        np.testing.assert_array_equal(nodes,np.arange(item['n_nodes']))
        result.append((item,nodes,x,extra))
    torch.backends.cuda.matmul.allow_tf32=False
    return result


def reader(plan,out=OUT):
    import lance
    tables={name:lance.dataset(info['uri'],version=info['version'],
        metadata_cache_size_bytes=64*1024**2,index_cache_size_bytes=64*1024**2)
        for name,info in plan['tables'].items() if name in ['named_skeletons','named_node_embeddings']}
    def scan(name,columns,clause):
        return tables[name].scanner(columns=columns,filter=clause,batch_readahead=1,
            fragment_readahead=2,io_buffer_size=64*1024**2).to_table()
    def load(items):
        start=time.perf_counter();teasar=[i for i in items if i['geometry']=='teasar'];embeddings={}
        if teasar:
            clause=' OR '.join(f"(root_id = {i['root_id']} AND skeleton_name IN ('{i['skeleton_name']}', '{i['embedding_skeleton_name']}'))" for i in teasar)
            geometry=scan('named_skeletons',['root_id','skeleton_name','coords'],clause)
            check_geometry(geometry,teasar)
            clause=f"run_id = '{EMBED_RUN}' AND checkpoint_id = '{EMBED_CKPT}' AND ("+' OR '.join(
                f"(root_id = {i['root_id']} AND skeleton_name = '{i['embedding_skeleton_name']}' AND node_id < {i['n_nodes']})" for i in teasar)+')'
            table=scan('named_node_embeddings',['root_id','node_id','embedding'],clause)
            embeddings=partition_embeddings(table,teasar,allow_missing=True)
            original_coords={}
            # Retain coordinates only for cells that need a backfill.
            for i in teasar:
                if len(embeddings[i['root_id']][0])!=i['n_nodes']:
                    index=next(k for k,(r,name) in enumerate(zip(geometry['root_id'].to_pylist(),geometry['skeleton_name'].to_pylist()))
                               if r==i['root_id'] and name==i['skeleton_name'])
                    original_coords[i['root_id']]=coords_from_arrow(geometry['coords'][index]).copy()
            del geometry,table
        loaded=[]
        caves=[i for i in items if i['geometry']=='cave']
        if caves:
            import h5py
            info=plan['cave_h5'];path=Path(info['path']);stat=path.stat()
            assert stat.st_size==info['size'] and stat.st_mtime_ns==info['mtime_ns']
            with h5py.File(path) as exported:
                for i in caves:
                    if not i['h5_ranges']:raise ValueError('CAVE cell missing from pinned export')
                    nodes=np.concatenate([exported['nodes'][a:b] for a,b in i['h5_ranges']]).astype(np.int32)
                    x=np.concatenate([exported['embeddings'][a:b] for a,b in i['h5_ranges']])
                    embeddings[i['root_id'],'cave']=unique_embedding_rows(nodes,x)
        for i in items:
            extra={}
            if i['geometry']=='presynaptic':
                nodes,x,coords,ids=read_points(i);extra=dict(positions_nm=coords,synapse_ids=ids)
            else:nodes,x=embeddings[(i['root_id'],'cave') if i['geometry']=='cave' else i['root_id']]
            if i['geometry']=='teasar' and len(nodes)!=i['n_nodes']:
                cached=out/'embedding_backfill'/'teasar'/f"{i['root_id']}.npz"
                if cached.exists():
                    with np.load(cached) as z:
                        assert int(z['root_id'])==i['root_id'] and str(z['skeleton_name'])==i['skeleton_name']
                        assert str(z['run_id'])==EMBED_RUN and str(z['checkpoint_id'])==EMBED_CKPT
                        nodes,x=merge_backfill(nodes,x,z['node_ids'],z['embeddings'],i['n_nodes'])
                missing=np.setdiff1d(np.arange(i['n_nodes'],dtype=np.int32),nodes)
                if len(missing):extra.update(_missing_node_ids=missing,_original_coords=original_coords[i['root_id']])
            else:np.testing.assert_array_equal(nodes,np.arange(i['n_nodes']))
            if x.shape!=(len(nodes),64) or not np.isfinite(x).all():raise ValueError('invalid embeddings')
            loaded.append((i,nodes,x,extra))
        return loaded,time.perf_counter()-start
    return load


def fold_predictor(models,classes):
    """Stack independent fold weights; vmap evaluates all five in one batched call."""
    import torch
    from torch.func import stack_module_state,functional_call
    params,buffers=stack_module_state(models)
    base=copy.deepcopy(models[0]).to('meta')
    def call(p,b,x):return functional_call(base,(p,b),(x,))
    apply=torch.vmap(call,in_dims=(0,0,None))
    orders=torch.tensor([[c.index(label) for label in HIERARCHY.level_classes[0]] for c in classes],
        device=next(models[0].parameters()).device)
    def predict(x):
        raw=apply(params,buffers,x)
        return raw.gather(2,orders[:,None,:].expand(raw.shape[0],raw.shape[1],-1))
    return predict


def classify(loaded,predict,device,batch_size=32768):
    import torch
    # Concatenation packs small cells without altering per-node features or averaging.
    sizes=[len(x) for _,_,x,_ in loaded];x=np.concatenate([x for _,_,x,_ in loaded])
    logits=np.empty((5,len(x),4),np.float32)
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            features=torch.from_numpy(np.ascontiguousarray(x[start:start+batch_size])).to(device)
            result=predict(features).cpu().numpy()
            if not np.isfinite(result).all():raise ValueError('nonfinite logits')
            logits[:,start:start+len(features)]=result
    return logits,sizes


def write_batch(out,loaded,logits,sizes,plan_hash):
    start=0
    for (i,nodes,_,extra),size in zip(loaded,sizes,strict=True):
        save_npz(destination(out,i),node_ids=nodes,logits=logits[:,start:start+size],
            root_id=i['root_id'],skeleton_name=i['skeleton_name'] or '',
            classes=np.asarray(HIERARCHY.level_classes[0]),plan_sha256=plan_hash,**extra)
        start+=size
    return sum(sizes)


def prefetched(executor,load,batches,depth=2):
    source=iter(batches);queue=deque()
    for _ in range(depth):
        batch=next(source,None)
        if batch is not None:queue.append(executor.submit(load,batch))
    while queue:
        value=queue.popleft().result()
        batch=next(source,None)
        if batch is not None:queue.append(executor.submit(load,batch))
        yield value


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--benchmark',action='store_true');p.add_argument('--limit-batches',type=int,default=0)
    a=p.parse_args();plan=json.loads((a.output/'plan.json').read_text());plan_hash=sha(a.output/'plan.json')
    if not torch.cuda.is_available() and not a.benchmark:raise RuntimeError('GPU required')
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    models=load_models(plan,device);predict=fold_predictor(models,[m['classes'] for m in plan['models']])
    items=pending_items(plan,a.output,a.rank)
    if a.benchmark:
        items=[i for i in plan['items'] if i['task_id']==0 and destination(a.output,i).exists()][:24]
    batches=list(chunks(items))
    if a.limit_batches:batches=batches[:a.limit_batches]
    engine=None
    def get_engine():
        nonlocal engine
        if engine is None:
            from scripts.generate_cave_embedding_augmentations import FastAugInference,CHECKPOINT
            if Path(CHECKPOINT).stem!=EMBED_CKPT:raise ValueError('backfill checkpoint mismatch')
            engine=FastAugInference('clean',0,0,batch_size=32,num_threads=8)
        return engine
    load=reader(plan,a.output);begin=time.perf_counter();nodes=0;load_seconds=0.;gpu_seconds=0.;checked=0
    try:
        with ThreadPoolExecutor(max_workers=2) as reads,ThreadPoolExecutor(max_workers=2) as writes:
            pending=deque()
            for loaded,read_time in prefetched(reads,load,batches):
                load_seconds+=read_time
                loaded=complete_embeddings(loaded,a.output,plan_hash,get_engine)
                t=time.perf_counter();logits,sizes=classify(loaded,predict,device)
                gpu_seconds+=time.perf_counter()-t;nodes+=sum(sizes)
                if a.benchmark:
                    offset=0
                    for i,_,_,_ in loaded:
                        size=i['n_nodes']
                        with np.load(destination(a.output,i)) as previous:
                            expected=previous['logits']
                        np.testing.assert_allclose(logits[:,offset:offset+size],expected,rtol=3e-5,atol=3e-5)
                        np.testing.assert_array_equal(logits[:,offset:offset+size].argmax(2),expected.argmax(2))
                        offset+=size;checked+=1
                else:
                    if len(pending)>=2:pending.popleft().result()
                    pending.append(writes.submit(write_batch,a.output,loaded,logits,sizes,plan_hash))
                print(json.dumps(dict(rank=a.rank,nodes=nodes,seconds=round(time.perf_counter()-begin,2),
                    last_batch_nodes=sum(sizes),last_batch_load_seconds=round(read_time,2),
                    gpu_seconds=round(gpu_seconds,2))),flush=True)
            for future in pending:future.result()
    except Exception as exc:
        atomic_json(a.output/'status'/f'fast_{a.rank}_failure.json',dict(error=repr(exc)))
        raise
    report=dict(rank=a.rank,benchmark=a.benchmark,device=device,nodes=nodes,checked_files=checked,
        seconds=time.perf_counter()-begin,load_seconds=load_seconds,gpu_seconds=gpu_seconds,batches=len(batches))
    atomic_json(a.output/'status'/f"{'fast_benchmark' if a.benchmark else 'fast_infer'}_{a.rank}.json",report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
