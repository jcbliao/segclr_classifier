"""Five-fold cell typing of axon-filtered incoming fragments; local outputs only.

Each row remains a distinct synapse. Pool phi(raw nodes) for CAVE n10;
use the center node for single_pre_post. Both use the exact post point vector.
Six-class scores are conditional local-head logits, with explicit joint
probabilities and head mappings. Never softmax six conditional logits together.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.dataset_lcpn import load_hierarchy
from gnn.model import WindowClassifier
from scripts.subcompartment_predict_all import atomic_json,sha,EMBED_RUN,EMBED_CKPT

DEFAULT=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2')
FAMILIES=('cave_n10','single_pre_post')


def checkpoint(family,classes,fold):
    base=ROOT/'results/presynaptic'
    if family=='cave_n10':
        base=base/f'cave_skeletons_pre_post/k10/conf0.7/{classes}_class/fold{fold}/pre_post';k=10
    else:
        base=base/'native_single_pre_post/scale16/k1/conf0.7'
        if classes=='three':base=base/'three_class'
        base=base/f'fold{fold}/pre_post';k=1
    return base/f'gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n{k}_mixed16_sampled_pre_post_fold{fold}/checkpoint_best.pt'


def load_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    args=json.loads(path.parent.with_suffix('.json').read_text())['args']
    manifest=Path(args.get('manifest') or Path(args['presynaptic_database'])/'manifest.json')
    hierarchy=load_hierarchy(json.loads(manifest.read_text()))
    model=WindowClassifier(ck['config'],hierarchy).eval()
    model.load_state_dict(ck['model_state'],strict=True)
    if model.config.architecture!='pointwise_mlp' or model.config.postsynaptic_dim!=64:
        raise ValueError('unexpected model architecture')
    if model.config.append_presynaptic_mean:raise ValueError('unexpected extra mean')
    return model,manifest


class BagPredictor(nn.Module):
    """Padded-bag equivalent of WindowClassifier, suitable for fold vmap."""
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,x,mask,post):
        h=self.model.pointwise_mlp(x)
        pooled=(h*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        hidden=torch.cat((pooled,post),dim=-1)
        head=self.model.cls_head
        if head.trunk is not None:hidden=head.trunk(hidden)
        return tuple(layer(hidden) for layer in head.heads)


def vectorized_predictor(models,device):
    wrappers=[BagPredictor(model).to(device).eval() for model in models]
    params,buffers=torch.func.stack_module_state(wrappers)
    template=copy.deepcopy(wrappers[0]).to('meta')
    def call(p,b,x,mask,post):
        return torch.func.functional_call(template,(p,b),(x,mask,post))
    mapped=torch.vmap(call,in_dims=(0,0,None,None,None))
    return lambda x,mask,post:mapped(params,buffers,x,mask,post)


def prepare(out,workers):
    torch.set_num_threads(1)
    def one(key):
        family,classes,fold=key;path=checkpoint(*key);model,manifest=load_checkpoint(path)
        return dict(family=family,classes=classes,fold=fold,path=str(path),sha256=sha(path),
            manifest=str(manifest),manifest_sha256=sha(manifest),config=asdict(model.config),
            level_classes=model.cls_head.hierarchy.level_classes,
            head_nodes=model.cls_head.nodes,
            hierarchy_id=f'casey_neuron_{classes}_class_v1',
            agg_spec_id=f'pointwise_mlp_mean_k{10 if family=="cave_n10" else 1}',
            aggregation=dict(method='pointwise_mlp_mean',k_nearest=10 if family=='cave_n10' else 1,
                             window_nm=None,selection='nearest connected geodesic skeleton nodes; center included'),
            classifier_run_id=f'casey_{classes}_{family}_fold{fold}_{sha(path)[:12]}')
    keys=[(family,classes,fold) for family in FAMILIES for classes in ('three','six') for fold in range(5)]
    with ThreadPoolExecutor(max_workers=workers) as pool:models=list(pool.map(one,keys))
    for family in FAMILIES:
        for classes,n in [('three',3),('six',6)]:
            group=[m for m in models if m['family']==family and m['classes']==classes]
            if any(m['level_classes']!=group[0]['level_classes'] or m['head_nodes']!=group[0]['head_nodes'] for m in group):
                raise ValueError('fold vocabularies differ')
            if len(group[0]['level_classes'][-1])!=n:raise ValueError('unexpected class count')
    from segclr_db.hierarchies import casey_hierarchies
    expected_hierarchies={h.hierarchy_id:h for h in casey_hierarchies()}
    for info in models:
        expected=expected_hierarchies[info['hierarchy_id']]
        if tuple(map(tuple,info['level_classes']))!=expected.level_classes:
            raise ValueError('checkpoint class order differs from registered Casey hierarchy')
    plan=dict(models=models,fragment_plan_sha256=sha(out/'plan.json'),
        embedding_run_id=EMBED_RUN,embedding_checkpoint_id=EMBED_CKPT,
        filter='per fold: ready presynaptic mean embedding has subcompartment argmax axon',
        scores='raw conditional local-head logits; explicit joint probabilities; top-down class indices',
        identity=dict(root_id='cell_root_id',subject_root_id='partner_root_id',node_id='anchor_node_id',synapse_id='synapse_id'),
        shared_database_predictions_written=False)
    dest=out/'cell_types/plan.json'
    if dest.exists() and json.loads(dest.read_text())!=plan:raise ValueError('model plan already frozen differently')
    atomic_json(dest,plan);print(json.dumps(plan),flush=True)


def padded_fragments(table):
    raw=table['fragment_embeddings'].combine_chunks();offset=np.asarray(raw.offsets)
    counts=np.diff(offset);values=np.asarray(raw.values.values).reshape(-1,64)
    if np.any(counts>10):raise ValueError('fragment larger than k10')
    ready=np.asarray(table['status'])=='ready'
    np.testing.assert_array_equal(counts,np.asarray(table['n_embeddings_used']))
    if np.any(ready&(counts<1)) or not np.isfinite(values).all():raise ValueError('invalid fragment features')
    padded=np.zeros((len(table),10,64),np.float32);mask=np.arange(10)[None,:]<counts[:,None]
    padded[mask]=values
    return padded,mask.astype(np.float32)


def post_features(out,rank,workers):
    import duckdb
    c=duckdb.connect();c.execute(f'SET threads={workers}');c.execute("SET memory_limit='12GB'")
    c.execute('SET temp_directory=?',[str(out/'cell_types'/f'tmp_{rank}')])
    cached=out/'cell_types/post_shards'/f'rank={rank}'
    if cached.exists():
        c.read_parquet(str(cached/'*.parquet'),hive_partitioning=False).create_view('post')
        table=c.sql('SELECT * FROM post ORDER BY synapse_id').arrow().read_all()
    else:
        c.read_parquet(str(out/'cohort'/f'{rank:03d}.parquet')).create_view('wanted')
        c.read_parquet(str(out/'postsynaptic_points/*.parquet')).create_view('post')
        table=c.sql('SELECT p.* FROM post p SEMI JOIN wanted w USING(cell_root_id,synapse_id) ORDER BY synapse_id').arrow().read_all()
    c.close();ids=np.asarray(table['synapse_id'])
    if np.any(ids[1:]<=ids[:-1]):raise ValueError('post synapse identities not unique')
    return table,ids


def prepare_posts(out,workers):
    """One parallel batched join, instead of rereading all points on every GPU."""
    import duckdb
    digest=sha(out/'plan.json');ready=out/'cell_types/post_shards_complete.json'
    if ready.exists():
        if json.loads(ready.read_text())['fragment_plan_sha256']!=digest:raise ValueError('stale post cache')
        return
    dest=out/'cell_types/post_shards';temp=out/'cell_types/post_shards.tmp'
    if dest.exists() or temp.exists():raise ValueError('incomplete post partition needs inspection')
    c=duckdb.connect();c.execute(f'SET threads={workers}');c.execute("SET memory_limit='36GB'")
    c.execute('SET temp_directory=?',[str(out/'cell_types/post_join_tmp')])
    c.read_parquet(str(out/'cohort/*.parquet'),filename=True).create_view('wanted_files')
    c.execute("CREATE VIEW wanted AS SELECT cell_root_id,synapse_id,CAST(regexp_extract(filename,'([0-9]{3})[.]parquet$',1) AS INTEGER) AS rank FROM wanted_files")
    c.read_parquet(str(out/'postsynaptic_points/*.parquet')).create_view('post')
    c.execute('CREATE VIEW paired AS SELECT p.*,w.rank FROM post p INNER JOIN wanted w USING(cell_root_id,synapse_id)')
    expected=json.loads((out/'plan.json').read_text())['synapses']
    rows,unique=c.sql('SELECT count(*),count(DISTINCT synapse_id) FROM paired').fetchone()
    if rows!=expected or rows!=unique:raise ValueError(f'post pairing incomplete: {rows}/{expected}, unique={unique}')
    # Structured relation export handles paths without SQL interpolation.
    c.sql('SELECT * FROM paired').write_parquet(str(temp),partition_by=['rank'],compression='zstd',row_group_size=32768)
    c.close();temp.rename(dest)
    atomic_json(ready,dict(rows=rows,fragment_plan_sha256=digest))
    print('partitioned postsynaptic inputs',rows,flush=True)


def nullable_vectors(values,included):
    values=np.asarray(values,np.float32)
    # Nullable fixed-size lists cannot round-trip through our Parquet reader.
    # Variable-size lists retain explicit nulls; the frozen class order fixes D.
    offsets=np.arange(len(values)+1,dtype=np.int32)*values.shape[1]
    return pa.ListArray.from_arrays(pa.array(offsets),pa.array(values.reshape(-1)),mask=pa.array(~included))


def writer_payload(table,info):
    """Convert one model/fold to SegCLRWriter.add_predictions arguments.

    This does not register anything or write to the shared database. The
    prediction run must record conditional-head score semantics from the plan.
    Uncertainty is one minus the joint probability of the top-down chosen class.
    """
    prefix=f"{info['family']}_{info['classes']}_fold{info['fold']}"
    table=table.filter(table[prefix+'_included']);n=len(table)
    labels=[];uncertainty=[];logits=[]
    for level,classes in enumerate(info['level_classes']):
        name=f'{prefix}_level{level}';index=np.asarray(table[name+'_class_index'],np.int64)
        scores=np.asarray(table[name+'_logits'].combine_chunks().values).reshape(n,len(classes))
        probs=np.asarray(table[name+'_probabilities'].combine_chunks().values).reshape(n,len(classes))
        labels.append(np.asarray(classes,object)[index])
        uncertainty.append(1-probs[np.arange(n),index]);logits.append(scores)
    return dict(root_ids=np.asarray(table['cell_root_id'],np.int64),
        subject_root_ids=np.asarray(table['partner_root_id'],np.int64),
        node_ids=np.asarray(table['anchor_node_id'],np.int32),
        synapse_ids=np.asarray(table['synapse_id'],np.int64),
        labels=np.column_stack(labels),uncertainty=np.column_stack(uncertainty),logits=logits,
        record_work=False)


def infer(out,rank,workers,device,batch_size,limit_parts):
    plan=json.loads((out/'cell_types/plan.json').read_text());digest=sha(out/'cell_types/plan.json')
    if sha(out/'plan.json')!=plan['fragment_plan_sha256']:raise ValueError('fragment plan changed')
    torch.set_num_threads(workers)
    groups={}
    # Independent checkpoint reads are parallel; GPU state construction follows.
    def load(info):
        path=Path(info['path'])
        if sha(path)!=info['sha256']:raise ValueError('checkpoint changed')
        model,manifest=load_checkpoint(path)
        if info.get('manifest_sha256') and sha(manifest)!=info['manifest_sha256']:raise ValueError('training manifest changed')
        if tuple(map(tuple,info['level_classes']))!=tuple(map(tuple,model.cls_head.hierarchy.level_classes)):
            raise ValueError('model vocabulary changed')
        if info.get('head_nodes') is not None and info['head_nodes']!=model.cls_head.nodes:
            raise ValueError('local-head mapping changed')
        return model
    with ThreadPoolExecutor(max_workers=workers) as pool:loaded=list(pool.map(load,plan['models']))
    for family in FAMILIES:
        for classes in ('three','six'):
            selected=[(info,model) for info,model in zip(plan['models'],loaded) if info['family']==family and info['classes']==classes]
            models=[m for _,m in selected]
            groups[(family,classes)]=(vectorized_predictor(models,device),models[0].cls_head,selected[0][0])
    post,post_ids=post_features(out,rank,workers)
    destdir=out/'cell_types/parts'/f'{rank:03d}';destdir.mkdir(parents=True,exist_ok=True)
    paths=sorted((out/'parts'/f'{rank:03d}').glob('[0-9][0-9][0-9][0-9][0-9].parquet'))
    if limit_parts:paths=paths[:limit_parts]
    stats=dict(rank=rank,rows=0,parts=0,plan_sha256=digest,included_by_fold=[0]*5);begin=time.monotonic()
    def process(path,table):
        dest=destdir/path.name
        if dest.exists():
            done=pq.read_table(dest)
            if done.schema.metadata.get(b'cell_type_plan_sha256')!=digest.encode():raise ValueError('stale predictions')
            return done
        if table.schema.metadata.get(b'plan_sha256')!=plan['fragment_plan_sha256'].encode():raise ValueError('stale fragments')
        syn=np.asarray(table['synapse_id']);index=np.searchsorted(post_ids,syn)
        if np.any(index>=len(post_ids)) or not np.array_equal(post_ids[index],syn):raise ValueError('missing postsynaptic embeddings')
        paired=post.take(pa.array(index))
        np.testing.assert_array_equal(np.asarray(paired['cell_root_id']),np.asarray(table['cell_root_id']))
        features,mask=padded_fragments(table)
        post_x=np.asarray(paired['post_embedding'].combine_chunks().values).reshape(-1,64)
        if not np.isfinite(post_x).all():raise ValueError('nonfinite postsynaptic embeddings')
        included=np.column_stack([np.asarray(table[f'presynaptic_fold{f}_is_axon']) for f in range(5)])
        selected=np.flatnonzero(included.any(1));n=len(table)
        result=table.drop(['fragment_embeddings','mean_embedding'])
        for name in paired.column_names:
            if name not in ('cell_root_id','synapse_id','post_embedding'):result=result.append_column(name,paired[name])
        for (family,classes),(predict,decoder,info) in groups.items():
            levels=info['level_classes']
            scores=[np.zeros((5,n,len(labels)),np.float32) for labels in levels]
            probs=[np.zeros_like(v) for v in scores];predictions=np.zeros((5,n,len(levels)),np.int16)
            with torch.inference_mode():
                for start in range(0,len(selected),batch_size):
                    idx=selected[start:start+batch_size];k=10 if family=='cave_n10' else 1
                    x=torch.from_numpy(features[idx,:k].copy()).to(device)
                    m=torch.from_numpy(mask[idx,:k].copy()).to(device)
                    px=torch.from_numpy(post_x[idx].copy()).to(device)
                    heads=predict(x,m,px);b=len(idx)
                    decoded=decoder.distribution_from_head_logits(tuple(h.reshape(5*b,-1) for h in heads))
                    for level in range(len(levels)):
                        scores[level][:,idx]=decoded['level_logits'][level].reshape(5,b,-1).cpu().numpy()
                        probs[level][:,idx]=decoded['level_probabilities'][level].reshape(5,b,-1).cpu().numpy()
                    predictions[:,idx]=decoded['predictions'].reshape(5,b,-1).cpu().numpy()
            for fold in range(5):
                prefix=f'{family}_{classes}_fold{fold}';keep=included[:,fold]
                result=result.append_column(prefix+'_included',pa.array(keep))
                used=mask.sum(1).astype(np.int16) if family=='cave_n10' else np.ones(n,np.int16)
                result=result.append_column(prefix+'_n_embeddings_used',pa.array(np.where(keep,used,0)))
                for level in range(len(levels)):
                    if keep.any():
                        if not np.isfinite(scores[level][fold,keep]).all():raise ValueError('nonfinite logits')
                        np.testing.assert_allclose(probs[level][fold,keep].sum(1),1,rtol=1e-5,atol=1e-6)
                    name=f'{prefix}_level{level}'
                    result=result.append_column(name+'_logits',nullable_vectors(scores[level][fold],keep))
                    result=result.append_column(name+'_probabilities',nullable_vectors(probs[level][fold],keep))
                    result=result.append_column(name+'_class_index',pa.array(predictions[fold,:,level],mask=~keep))
        result=result.replace_schema_metadata({b'cell_type_plan_sha256':digest.encode(),b'fragment_plan_sha256':plan['fragment_plan_sha256'].encode()})
        tmp=dest.with_suffix(f'.tmp.{os.getpid()}');pq.write_table(result,tmp,compression='zstd');tmp.replace(dest)
        return result
    # One read in flight while the current batch executes; bounded memory.
    with ThreadPoolExecutor(max_workers=2) as pool:
        future=pool.submit(pq.read_table,paths[0]) if paths else None
        for i,path in enumerate(paths):
            table=future.result()
            future=pool.submit(pq.read_table,paths[i+1]) if i+1<len(paths) else None
            done=process(path,table);stats['rows']+=len(done);stats['parts']+=1
            for f in range(5):stats['included_by_fold'][f]+=int(np.asarray(done[f'presynaptic_fold{f}_is_axon']).sum())
            stats['seconds']=time.monotonic()-begin
            atomic_json(out/'cell_types/status'/f'{rank:03d}.json',stats);print(json.dumps(stats),flush=True)
    if not limit_parts:
        expected=json.loads((out/'status'/f'{rank:03d}_complete.json').read_text())
        if stats['rows']!=expected['synapses'] or stats['parts']!=expected['parts']:raise ValueError('incomplete inference shard')
        atomic_json(out/'cell_types/status'/f'{rank:03d}_complete.json',stats)


def finalize(out,workers):
    import duckdb
    fragment=json.loads((out/'summary.json').read_text());digest=sha(out/'cell_types/plan.json')
    reports=[json.loads((out/'cell_types/status'/f'{rank:03d}_complete.json').read_text()) for rank in range(32)]
    if any(r['plan_sha256']!=digest for r in reports):raise ValueError('stale shard')
    if sum(r['rows'] for r in reports)!=fragment['synapses']:raise ValueError('incomplete predictions')
    c=duckdb.connect(str(out/'cell_types/predictions.duckdb'));c.execute(f'SET threads={workers}')
    c.read_parquet(str(out/'cell_types/parts/*/[0-9][0-9][0-9][0-9][0-9].parquet')).create_view('predictions',replace=True)
    rows,distinct=c.sql('SELECT count(*),count(DISTINCT synapse_id) FROM predictions').fetchone()
    if rows!=fragment['synapses'] or rows!=distinct:raise ValueError('lost or duplicated synapses')
    counts={}
    score_checks=[]
    model_plan=json.loads((out/'cell_types/plan.json').read_text())
    for family in FAMILIES:
        for classes in ('three','six'):
            for fold in range(5):
                prefix=f'{family}_{classes}_fold{fold}'
                c.execute(f'CREATE OR REPLACE VIEW {prefix} AS SELECT * FROM predictions WHERE {prefix}_included')
                counts[prefix]=c.sql(f'SELECT count(*) FROM {prefix}').fetchone()[0]
                if counts[prefix]!=fragment['axon_counts_by_fold'][str(fold)]:raise ValueError('filter counts differ')
                expected_count='n_embeddings_used' if family=='cave_n10' else '1'
                score_checks.append(f'count(*) FILTER (WHERE ({prefix}_included AND {prefix}_n_embeddings_used!={expected_count}) OR (NOT {prefix}_included AND {prefix}_n_embeddings_used!=0))')
                info=next(m for m in model_plan['models'] if m['family']==family and m['classes']==classes and m['fold']==fold)
                for level,labels in enumerate(info['level_classes']):
                    col=f'{prefix}_level{level}';keep=f'{prefix}_included';dim=len(labels)
                    bad=f'''({keep} AND ({col}_probabilities IS NULL OR {col}_logits IS NULL OR {col}_class_index IS NULL
                        OR len({col}_probabilities)!={dim} OR len({col}_logits)!={dim}
                        OR NOT isfinite(list_sum({col}_probabilities))
                        OR abs(list_sum({col}_probabilities)-1)>0.0001
                        OR list_min({col}_probabilities)<-0.00001 OR list_max({col}_probabilities)>1.00001
                        OR NOT isfinite(list_sum({col}_logits))
                        OR {col}_class_index<0 OR {col}_class_index>={dim}))
                        OR (NOT {keep} AND ({col}_probabilities IS NOT NULL OR {col}_logits IS NOT NULL OR {col}_class_index IS NOT NULL))'''
                    score_checks.append(f'count(*) FILTER (WHERE {bad})')
    invalid=c.sql('SELECT '+','.join(score_checks)+' FROM predictions').fetchone()
    if any(invalid):raise ValueError(f'invalid probability/logit/class/mask columns: {invalid}')
    c.sql('SELECT n_embeddings_used,count(*) AS synapses FROM predictions GROUP BY n_embeddings_used ORDER BY n_embeddings_used').df().to_csv(out/'cell_types/embedding_count_distribution.csv',index=False)
    c.close();summary=dict(rows=rows,distinct_synapses=distinct,models=20,counts=counts,plan_sha256=digest,
        score_columns_validated=True,shared_database_predictions_written=False)
    atomic_json(out/'cell_types/summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','prepare-posts','infer','finalize'])
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8192)
    p.add_argument('--limit-parts',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':prepare(a.output,a.workers)
    elif a.stage=='prepare-posts':prepare_posts(a.output,a.workers)
    elif a.stage=='infer':infer(a.output,a.rank,a.workers,a.device,a.batch_size,a.limit_parts)
    else:finalize(a.output,a.workers)


if __name__=='__main__':main()
